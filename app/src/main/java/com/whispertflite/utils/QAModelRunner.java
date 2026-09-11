package com.whispertflite.utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Exécute le modèle QA (extraction_qa_model.tflite) : prend une question et
 * un contexte (le texte transcrit), et retourne le passage de texte qui
 * répond à la question -- exactement comme le modèle a été entraîné sur
 * Colab (contexte/question/réponse).
 */
public class QAModelRunner {
    private static final String TAG = "QAModelRunner";
    private static final String MODEL_NAME = "extraction_qa_model.tflite";
    private static final String VOCAB_NAME = "vocab_camembert.json";
    private static final int MAX_LENGTH = 256;

    private final Interpreter interpreter;
    private final SentencePieceTokenizer tokenizer;

    public QAModelRunner(Context context) throws IOException {
        copyModelIfNeeded(context);
        this.interpreter = new Interpreter(loadModelFile(context));
        this.tokenizer = new SentencePieceTokenizer(context, VOCAB_NAME);
    }

    private void copyModelIfNeeded(Context context) throws IOException {
        File modelFile = new File(context.getFilesDir(), MODEL_NAME);
        if (!modelFile.exists()) {
            AssetManager assetManager = context.getAssets();
            try (InputStream in = assetManager.open(MODEL_NAME);
                 FileOutputStream out = new FileOutputStream(modelFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
                Log.d(TAG, "Modèle QA copié : " + modelFile.getAbsolutePath());
            }
        }
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        File modelFile = new File(context.getFilesDir(), MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(modelFile);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }

    /**
     * Pose une question sur un texte (contexte) et retourne le passage de
     * texte trouvé par le modèle. Retourne une chaîne vide si rien de
     * pertinent n'est trouvé.
     */
    public String answer(String context, String question) {
        // Tokeniser la question
        List<Integer> questionIds = tokenizer.encode(question, null);

        // Tokeniser le contexte, en gardant les offsets (position dans
        // le texte original) pour pouvoir reconstruire la réponse.
        List<int[]> contextOffsets = new ArrayList<>();
        List<Integer> contextIds = tokenizer.encode(context, contextOffsets);

        // Construire la séquence complète
        int bos = tokenizer.getBosId();
        int eos = tokenizer.getEosId();

        // On réserve 4 tokens spéciaux
        int maxContextTokens = MAX_LENGTH - questionIds.size() - 4; // tokens spéciaux
        if (maxContextTokens < 1) {
            maxContextTokens = 1;
        }
        int contextTokenCount = Math.min(contextIds.size(), maxContextTokens);

        List<Integer> inputIds = new ArrayList<>();
        inputIds.add(bos);
        inputIds.addAll(questionIds);
        inputIds.add(eos);
        inputIds.add(eos);

        int contextStartIndexInSequence = inputIds.size(); // position du 1er token du contexte
        for (int i = 0; i < contextTokenCount; i++) {
            inputIds.add(contextIds.get(i));
        }
        inputIds.add(eos);
        int contextEndIndexInSequence = inputIds.size() - 1; // dernier index valide du contexte

        //Préparer les tenseurs d'entrée (remplis à MAX_LENGTH avec du padding)
        int[][] ids = new int[1][MAX_LENGTH];
        int[][] mask = new int[1][MAX_LENGTH];
        int[][] segmentIds = new int[1][MAX_LENGTH];

        for (int i = 0; i < MAX_LENGTH; i++) {
            if (i < inputIds.size()) {
                ids[0][i] = inputIds.get(i);
                mask[0][i] = 1;
            } else {
                ids[0][i] = 0;
                mask[0][i] = 0;
            }
            segmentIds[0][i] = 0;
        }

        // Lancer l'inférence
        float[][] startLogits = new float[1][MAX_LENGTH];
        float[][] endLogits = new float[1][MAX_LENGTH];

        Object[] inputs = new Object[]{ids, mask, segmentIds};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, startLogits);
        outputs.put(1, endLogits);

        interpreter.runForMultipleInputsOutputs(inputs, outputs);

        // Trouver les meilleures positions de début/fin
        int bestStart = contextStartIndexInSequence;
        int bestEnd = contextStartIndexInSequence;
        float bestScore = Float.NEGATIVE_INFINITY;

        // Recherche la meilleure paire (début, fin) UNIQUEMENT dans la zone du
        // contexte (jamais dans la question), pour éviter que le modèle
        // "réponde" avec un morceau de la question elle-même.

        for (int start = contextStartIndexInSequence; start <= contextEndIndexInSequence; start++) {
            for (int end = start; end <= contextEndIndexInSequence; end++) {
                float score = startLogits[0][start] + endLogits[0][end];
                if (score > bestScore) {
                    bestScore = score;
                    bestStart = start;
                    bestEnd = end;
                }
            }
        }

        // Convertir les positions de tokens en positions de caractères
        // dans le texte original du contexte, via les offsets calculés.
        int contextTokenStart = bestStart - contextStartIndexInSequence;
        int contextTokenEnd = bestEnd - contextStartIndexInSequence;

        if (contextTokenStart < 0 || contextTokenEnd >= contextOffsets.size()
                || contextTokenStart > contextTokenEnd) {
            return "";
        }

        int charStart = contextOffsets.get(contextTokenStart)[0];
        int charEnd = contextOffsets.get(contextTokenEnd)[1];

        // Les offsets sont calculés sur le texte normalisé (avec '▁' pour
        // les espaces) ; on reconstruit la réponse en repartant du texte
        // normalisé puis en remettant de vrais espaces.
        String normalizedContext = "\u2581" + context.trim().replaceAll("\\s+", " ").replace(" ", "\u2581");
        if (charStart < 0 || charEnd > normalizedContext.length() || charStart >= charEnd) {
            return "";
        }
        String rawAnswer = normalizedContext.substring(charStart, charEnd);
        return rawAnswer.replace("\u2581", " ").trim();
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
