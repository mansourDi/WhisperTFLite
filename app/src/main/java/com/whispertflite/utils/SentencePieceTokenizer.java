package com.whispertflite.utils;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tokenizer SentencePiece (algorithme "Unigram", celui utilisé par CamemBERT)
 * réécrit en Java pur, sans bibliothèque externe.
 *
 * Principe : découpe un texte en "morceaux" (tokens) connus du vocabulaire,
 * en choisissant la découpe qui maximise la somme des scores (algorithme de
 * Viterbi), exactement comme le fait la bibliothèque Python `sentencepiece`.
 */
public class SentencePieceTokenizer {

    private static final String SPACE_MARKER = "\u2581"; // le caractère '▁'
    private static final int UNK_ID = 0;
    private static final int BOS_ID = 1;  // <s>
    private static final int EOS_ID = 2;  // </s>
    private static final double UNK_PENALTY = -20.0; // pénalité pour un caractère inconnu

    private final Map<String, Integer> pieceToId = new HashMap<>();
    private final Map<String, Double> pieceToScore = new HashMap<>();
    private final Map<Integer, String> idToPiece = new HashMap<>();
    private int maxPieceLength = 1;

    /**
     * Charge le vocabulaire depuis le fichier JSON (vocab_camembert.json)
     * placé dans les assets de l'application.
     */
    public SentencePieceTokenizer(Context context, String assetFileName) throws IOException {
        StringBuilder jsonBuilder = new StringBuilder();
        InputStream is = context.getAssets().open(assetFileName);
        BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
        String line;
        while ((line = reader.readLine()) != null) {
            jsonBuilder.append(line);
        }
        reader.close();

        try {
            JSONArray arr = new JSONArray(jsonBuilder.toString());
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                String piece = obj.getString("piece");
                double score = obj.getDouble("score");
                int id = obj.getInt("id");

                pieceToId.put(piece, id);
                pieceToScore.put(piece, score);
                idToPiece.put(id, piece);

                if (piece.length() > maxPieceLength) {
                    maxPieceLength = piece.length();
                }
            }
        } catch (Exception e) {
            throw new IOException("Erreur de parsing du vocabulaire : " + e.getMessage());
        }
    }

    /**
     * Normalise le texte comme le fait SentencePiece : NFKC, remplace les
     * espaces par le marqueur '▁', et préfixe le texte par '▁'.
     */
    private String normalize(String text) {
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC);
        normalized = normalized.trim().replaceAll("\\s+", " ");
        normalized = normalized.replace(" ", SPACE_MARKER);
        if (!normalized.startsWith(SPACE_MARKER)) {
            normalized = SPACE_MARKER + normalized;
        }
        return normalized;
    }

    /**
     * Découpe le texte normalisé en tokens, en cherchant la meilleure
     * segmentation possible (score total maximal) via programmation
     * dynamique (algorithme de Viterbi), comme le fait SentencePiece.
     *
     * Retourne la liste des ids de tokens, ainsi que (via offsets) la
     * position de fin (en caractères, dans le texte normalisé) de chaque
     * token, utile pour retrouver le texte original de la réponse plus tard.
     */
    public List<Integer> encode(String text, List<int[]> offsetsOut) {
        String normalized = normalize(text);
        int n = normalized.length();

        double[] dp = new double[n + 1];
        int[] backPos = new int[n + 1];   // position de départ du meilleur token arrivant en i
        int[] backId = new int[n + 1];    // id du meilleur token arrivant en i

        // Algorithme de Viterbi : pour chaque position du texte, on calcule le
        // meilleur score cumulé possible pour l'atteindre, en testant tous les
        // tokens du vocabulaire pouvant s'y terminer (du plus long au plus court)
        for (int i = 1; i <= n; i++) {
            dp[i] = Double.NEGATIVE_INFINITY;
        }

        for (int i = 0; i < n; i++) {
            if (dp[i] == Double.NEGATIVE_INFINITY && i != 0) {
                continue;
            }
            int maxLen = Math.min(maxPieceLength, n - i);
            boolean foundAny = false;
            for (int len = maxLen; len >= 1; len--) {
                String candidate = normalized.substring(i, i + len);
                Double score = pieceToScore.get(candidate);
                if (score != null) {
                    double newScore = dp[i] + score;
                    if (newScore > dp[i + len]) {
                        dp[i + len] = newScore;
                        backPos[i + len] = i;
                        backId[i + len] = pieceToId.get(candidate);
                        foundAny = true;
                    }
                }
            }
            if (!foundAny) {
                double newScore = dp[i] + UNK_PENALTY;
                if (newScore > dp[i + 1]) {
                    dp[i + 1] = newScore;
                    backPos[i + 1] = i;
                    backId[i + 1] = UNK_ID;
                }
            }
        }

        List<Integer> idsReversed = new ArrayList<>();
        List<int[]> offsetsReversed = new ArrayList<>();
        int pos = n;
        while (pos > 0) {
            int start = backPos[pos];
            int id = backId[pos];
            idsReversed.add(id);
            offsetsReversed.add(new int[]{start, pos});
            pos = start;
        }

        List<Integer> ids = new ArrayList<>();
        for (int i = idsReversed.size() - 1; i >= 0; i--) {
            ids.add(idsReversed.get(i));
            if (offsetsOut != null) {
                offsetsOut.add(offsetsReversed.get(i));
            }
        }
        return ids;
    }

    /**
     * Reconstruit le texte à partir d'une liste d'ids de tokens (utile pour
     * décoder la réponse extraite par le modèle).
     */
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String piece = idToPiece.get(id);
            if (piece == null || id == BOS_ID || id == EOS_ID) {
                continue;
            }
            sb.append(piece);
        }
        return sb.toString().replace(SPACE_MARKER, " ").trim();
    }

    public int getBosId() {
        return BOS_ID;
    }

    public int getEosId() {
        return EOS_ID;
    }
}