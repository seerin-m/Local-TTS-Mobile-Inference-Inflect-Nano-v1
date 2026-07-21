package com.mobilerun.inflecttts;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import ai.onnxruntime.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.*;
import java.nio.ByteBuffer;
public class MainActivity extends AppCompatActivity {

    private static final String LAPTOP_IP = "your laptop ip"; // your laptop IP
    private static final int PORT = 8765;
    private static final int SAMPLE_RATE = 24000;
    private static final int ABS_FRAME_BINS = 512;
    private static final int MAX_FRAMES = 1400;

    private OrtEnvironment env;
    private OrtSession encoderSession, decoderSession, vocoderSession;
    private EditText etText;
    private TextView tvStatus, tvLatency;
    private Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etText = findViewById(R.id.etText);
        tvStatus = findViewById(R.id.tvStatus);
        tvLatency = findViewById(R.id.tvLatency);
        Button btnSpeak = findViewById(R.id.btnSpeak);

        tvStatus.setText("Loading models...");
        new Thread(this::loadModels).start();

        btnSpeak.setOnClickListener(v -> {
            String text = etText.getText().toString().trim();
            if (text.isEmpty()) return;
            tvStatus.setText("Generating...");
            tvLatency.setText("");
            new Thread(() -> runTTS(text)).start();
        });
    }

    private void loadModels() {
        try {
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setIntraOpNumThreads(4);
            encoderSession = env.createSession(loadAsset("acoustic_encoder.onnx"), opts);
            decoderSession  = env.createSession(loadAsset("acoustic_decoder.onnx"), opts);
            vocoderSession  = env.createSession(loadAsset("vocoder.onnx"), opts);
            mainHandler.post(() -> tvStatus.setText("Models loaded. Ready."));
        } catch (Exception e) {
            mainHandler.post(() -> tvStatus.setText("Model load error: " + e.getMessage()));
        }
    }

    private byte[] loadAsset(String name) throws IOException {
        InputStream is = getAssets().open(name);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        return bos.toByteArray();
    }

    private void runTTS(String text) {
        try {
            long t0 = System.currentTimeMillis();

            // Step 1: phonemize via laptop
            mainHandler.post(() -> tvStatus.setText("Phonemizing..."));
            JSONObject phonemes = phonemize(text);
            long[] phone   = toArray(phonemes.getJSONArray("phone"));
            long[] tone    = toArray(phonemes.getJSONArray("tone"));
            long[] lang    = toArray(phonemes.getJSONArray("lang"));
            long[] speaker = toArray(phonemes.getJSONArray("speaker"));
            int T = phone.length;

            // Step 2: encoder
            mainHandler.post(() -> tvStatus.setText("Running encoder..."));
            Map<String, OnnxTensor> encInputs = new HashMap<>();
            encInputs.put("phone",   OnnxTensor.createTensor(env, LongBuffer.wrap(phone),   new long[]{1, T}));
            encInputs.put("tone",    OnnxTensor.createTensor(env, LongBuffer.wrap(tone),    new long[]{1, T}));
            encInputs.put("lang",    OnnxTensor.createTensor(env, LongBuffer.wrap(lang),    new long[]{1, T}));
            encInputs.put("speaker", OnnxTensor.createTensor(env, LongBuffer.wrap(speaker), new long[]{1}));
            OrtSession.Result encOut = encoderSession.run(encInputs);
            float[][][] conditioned = (float[][][]) encOut.get(0).getValue();
// durations may come as long or float
            Object durRaw = encOut.get(1).getValue();
            float[][] durations;
            if (durRaw instanceof long[][]) {
                long[][] durLong = (long[][]) durRaw;
                durations = new float[durLong.length][durLong[0].length];
                for (int i = 0; i < durLong.length; i++)
                    for (int j = 0; j < durLong[i].length; j++)
                        durations[i][j] = (float) durLong[i][j];
            } else {
                durations = (float[][]) durRaw;
            }
            Object pitchObj = encOut.get(2).getValue();

            float[] pitch1d;

            if (pitchObj instanceof float[][][]) {
                float[][][] p = (float[][][]) pitchObj;
                pitch1d = new float[p[0].length];

                for (int i = 0; i < p[0].length; i++) {
                    pitch1d[i] = p[0][i][0];
                }
            } else {
                float[][] p = (float[][]) pitchObj;
                pitch1d = p[0];
            }

            Map<String, OnnxTensor> decInputs =
                    hostRegulate(conditioned[0], durations[0], pitch1d);
            // Step 4: decoder
            mainHandler.post(() -> tvStatus.setText("Running decoder..."));
            OrtSession.Result decOut = decoderSession.run(decInputs);
            Object melRaw = decOut.get(0).getValue();

// normalize mel to float[] regardless of output shape
            float[] melFlat;
            int melF, melC;
            if (melRaw instanceof float[][][]) {
                float[][][] mel = (float[][][]) melRaw;
                melF = mel[0].length;
                melC = mel[0][0].length;
                melFlat = new float[melF * melC];
                for (int i = 0; i < melF; i++)
                    for (int j = 0; j < melC; j++)
                        melFlat[i * melC + j] = mel[0][i][j];
            } else if (melRaw instanceof float[][]) {
                float[][] mel = (float[][]) melRaw;
                melF = mel.length;
                melC = mel[0].length;
                melFlat = new float[melF * melC];
                for (int i = 0; i < melF; i++)
                    System.arraycopy(mel[i], 0, melFlat, i * melC, melC);
            } else {
                melFlat = (float[]) melRaw;
                melF = melFlat.length;
                melC = 1;
            }
            Map<String, OnnxTensor> vocInputs = new HashMap<>();
            vocInputs.put("mel", OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(melFlat), new long[]{1, melF, melC}));
            OrtSession.Result vocOut = vocoderSession.run(vocInputs);
            float[] wav = flattenFloat(vocOut.get(0).getValue());

            long inferMs = System.currentTimeMillis() - t0;
            float audioDur = wav.length / (float) SAMPLE_RATE;
            float rtf = (inferMs / 1000f) / audioDur;

            // Step 6: play audio
            mainHandler.post(() -> tvStatus.setText("Playing..."));
            playAudio(wav);

            mainHandler.post(() -> {
                tvStatus.setText("Done!");
                tvLatency.setText(String.format(
                        "Inference: %dms | Audio: %.2fs | RTF: %.3f", inferMs, audioDur, rtf));
            });

        } catch (Exception e) {
            mainHandler.post(() -> tvStatus.setText("Error: " + e.getMessage()));
            e.printStackTrace();
        }
    }

    private JSONObject phonemize(String text) throws Exception {
        URL url = new URL("http://" + LAPTOP_IP + ":" + PORT + "/phonemize");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        JSONObject body = new JSONObject();
        body.put("text", text);
        conn.getOutputStream().write(body.toString().getBytes());
        BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) sb.append(line);
        return new JSONObject(sb.toString());
    }

    private long[] toArray(JSONArray arr) throws Exception {
        long[] out = new long[arr.length()];
        for (int i = 0; i < arr.length(); i++) out[i] = arr.getLong(i);
        return out;
    }

    private Map<String, OnnxTensor> hostRegulate(float[][] c, float[] durRaw, float[] pitchRaw) throws OrtException {
        int T = c.length, H = c[0].length;
        int[] d = new int[T];
        int F = 0;
        for (int i = 0; i < T; i++) {
            d[i] = Math.max(0, Math.round(durRaw[i]));
            F += d[i];
        }
        float[][] frames = new float[F][H];
        float[] pitchFrame = new float[F];
        float[] rel = new float[F], relInv = new float[F], center = new float[F];
        float[] sinRel = new float[F], cosRel = new float[F];
        float[] tokenPos = new float[F], logDur = new float[F], dpf = new float[F];
        float[][] localCtx = new float[F][H * 3];
        long[] absPos = new long[F];

        int tc = 0;
        for (int i = 0; i < T; i++) if (d[i] > 0) tc++;
        int pos = 0;
        int[] starts = new int[T];
        for (int i = 0; i < T; i++) { starts[i] = pos; pos += d[i]; }

        pos = 0;
        for (int i = 0; i < T; i++) {
            float[] prev = i > 0 ? c[i-1] : c[0];
            float[] curr = c[i];
            float[] next = i < T-1 ? c[i+1] : c[T-1];
            for (int f = 0; f < d[i]; f++) {
                int fi = pos + f;
                System.arraycopy(curr, 0, frames[fi], 0, H);
                pitchFrame[fi] = i < pitchRaw.length ? pitchRaw[i] : 0;
                float r = d[i] > 1 ? (float)f / (d[i]-1) : 0;
                rel[fi] = r; relInv[fi] = 1-r; center[fi] = 1 - Math.abs(r*2-1);
                sinRel[fi] = (float)Math.sin(r * Math.PI);
                cosRel[fi] = (float)Math.cos(r * Math.PI);
                tokenPos[fi] = tc > 1 ? (float)i / (tc-1) : 0;
                logDur[fi] = (float)(Math.log1p(d[i]) / 6.0);
                dpf[fi] = d[i] / 40f;
                System.arraycopy(prev, 0, localCtx[fi], 0, H);
                System.arraycopy(curr, 0, localCtx[fi], H, H);
                System.arraycopy(next, 0, localCtx[fi], H*2, H);
                absPos[fi] = Math.min((long)fi * ABS_FRAME_BINS / Math.max(1, MAX_FRAMES), ABS_FRAME_BINS-1);
            }
            pos += d[i];
        }

        // flatten 2D arrays
        float[] framesFlat = new float[F * H];
        float[] frameMeta = new float[F * 8];
        float[] localCtxFlat = new float[F * H * 3];
        for (int i = 0; i < F; i++) {
            System.arraycopy(frames[i], 0, framesFlat, i*H, H);
            frameMeta[i*8]   = rel[i];    frameMeta[i*8+1] = relInv[i];
            frameMeta[i*8+2] = center[i]; frameMeta[i*8+3] = sinRel[i];
            frameMeta[i*8+4] = cosRel[i]; frameMeta[i*8+5] = tokenPos[i];
            frameMeta[i*8+6] = logDur[i]; frameMeta[i*8+7] = dpf[i];
            System.arraycopy(localCtx[i], 0, localCtxFlat, i*H*3, H*3);
        }
        boolean[] mask = new boolean[F];
        Arrays.fill(mask, true);

        Map<String, OnnxTensor> feeds = new HashMap<>();
        feeds.put("frames",        OnnxTensor.createTensor(env, FloatBuffer.wrap(framesFlat),   new long[]{1, F, H}));
        feeds.put("frame_meta",    OnnxTensor.createTensor(env, FloatBuffer.wrap(frameMeta),    new long[]{1, F, 8}));
        feeds.put("local_ctx_raw", OnnxTensor.createTensor(env, FloatBuffer.wrap(localCtxFlat), new long[]{1, F, H*3}));
        feeds.put("abs_pos",       OnnxTensor.createTensor(env, LongBuffer.wrap(absPos),        new long[]{1, F}));
        float[] pitchFrame2D = new float[F * 2];

        for (int i = 0; i < F; i++) {
            pitchFrame2D[i * 2] = pitchFrame[i];
            pitchFrame2D[i * 2 + 1] = 0f;
        }

        feeds.put(
                "pitch_frame",
                OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(pitchFrame2D),
                        new long[]{1, F, 2}
                )
        );
        // frame_mask as bool tensor
        boolean[][] frameMask = new boolean[1][F];

        for (int i = 0; i < F; i++) {
            frameMask[0][i] = true;
        }

        feeds.put("frame_mask",
                OnnxTensor.createTensor(env, frameMask));
        return feeds;
    }

    private float[] flattenFloat(Object val) {
        if (val instanceof float[]) return (float[]) val;
        if (val instanceof float[][]) {
            float[][] v = (float[][]) val;
            float[] out = new float[v[0].length];
            System.arraycopy(v[0], 0, out, 0, out.length);
            return out;
        }
        if (val instanceof float[][][]) {
            float[][][] v = (float[][][]) val;
            float[] out = new float[v[0][0].length];
            System.arraycopy(v[0][0], 0, out, 0, out.length);
            return out;
        }
        return new float[0];
    }

    private void playAudio(float[] wav) {
        short[] pcm = new short[wav.length];
        for (int i = 0; i < wav.length; i++)
            pcm[i] = (short) Math.max(-32768, Math.min(32767, (int)(wav[i] * 32767)));
        AudioTrack track = new AudioTrack(AudioManager.STREAM_MUSIC, SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
                pcm.length * 2, AudioTrack.MODE_STATIC);
        track.write(pcm, 0, pcm.length);
        track.play();
    }
}