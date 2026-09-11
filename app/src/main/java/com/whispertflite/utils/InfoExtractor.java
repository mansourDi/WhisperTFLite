package com.whispertflite.utils;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class InfoExtractor {
    private static final String TAG = "InfoExtractor";

    private final QAModelRunner qaModelRunner;

    // Expressions régulières pour extraire les constantes médicales
    // (inchangé -- on garde ce qui fonctionnait déjà)
    private static final Pattern TEMPERATURE_PATTERN = Pattern.compile("(\\d{2}\\.\\d)\\s?°C");
    private static final Pattern BLOOD_PRESSURE_PATTERN = Pattern.compile("(\\d{2,3})/(\\d{2,3})\\s?mmHg");
    private static final Pattern HEART_RATE_PATTERN = Pattern.compile("(\\d{2,3})\\s?beats per minute");
    private static final Pattern RESPIRATORY_RATE_PATTERN = Pattern.compile("(\\d{2})\\s?cycles per minute");
    private static final Pattern OXYGEN_SATURATION_PATTERN = Pattern.compile("(\\d{2})\\s?%");

    public InfoExtractor(Context context) throws IOException {
        this.qaModelRunner = new QAModelRunner(context);
    }

     // Extrait les champs texte (motif de consultation, histoire de la
     // maladie) à partir du texte transcrit, en utilisant réellement le
     // modèle QA entraîné.
    public Map<String, String> extractTextFields(String text) {
        Map<String, String> result = new HashMap<>();

        try {
            String motif = qaModelRunner.answer(text, "Quel est le motif de consultation ?");
            result.put("motif_consultation", motif);

            String histoire = qaModelRunner.answer(text, "Quelle est l'histoire de la maladie ?");
            result.put("histoire_maladie", histoire);
        } catch (Exception e) {
            Log.e(TAG, "Erreur lors de l'extraction QA : " + e.getMessage(), e);
        }

        return result;
    }

    // Extrait le motif de consultation et l'histoire de la maladie à partir
    // du texte transcrit, en interrogeant le modèle QA une fois par champ
    public Map<String, Float> extractMedicalData(String text) {
        Map<String, Float> extractedData = new HashMap<>();

        extractedData.put("temperature", extractValue(text, TEMPERATURE_PATTERN));
        extractedData.put("blood_pressure", extractBloodPressure(text));
        extractedData.put("heart_rate", extractValue(text, HEART_RATE_PATTERN));
        extractedData.put("respiratory_rate", extractValue(text, RESPIRATORY_RATE_PATTERN));
        extractedData.put("oxygen_saturation", extractValue(text, OXYGEN_SATURATION_PATTERN));

        return extractedData;
    }

    private float extractValue(String text, Pattern pattern) {
        Matcher matcher = pattern.matcher(text);
        if (matcher.find()) {
            return Float.parseFloat(matcher.group(1));
        }
        return 0.0f;
    }

    private Float extractBloodPressure(String text) {
        Pattern bpPattern = Pattern.compile("(\\d{2,3})[\\s/-](\\d{2,3})");
        Matcher matcher = bpPattern.matcher(text);

        if (matcher.find()) {
            int systolic = Integer.parseInt(matcher.group(1));
            int diastolic = Integer.parseInt(matcher.group(2));
            return (float) (systolic * 100 + diastolic);
        }
        return 0f;
    }

     // Ferme l'interpréteur TensorFlow Lite pour libérer les ressources.

    public void close() {
        if (qaModelRunner != null) {
            qaModelRunner.close();
        }
    }
}