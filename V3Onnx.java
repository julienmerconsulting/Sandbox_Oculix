import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * OmniParser icon_detect_v3 (YOLOv9-E, MIT) run through OpenCV dnn inside the JVM, Apertix only.
 *
 * java -cp oculixapi-4.0.0-complete-win.jar V3Onnx.java image.png [conf] [model.onnx]
 * (the Apertix jar alone is not enough: its loader wants JNA on the classpath)
 *
 * Same pipeline as icon_detect_v3_on_image.py: letterbox 640 on grey 114, six raw heads
 * (logits + distances at strides 8/16/32), sigmoid, box = anchor -/+ distance * stride,
 * conf filter, NMS IoU 0.7, unpad, "keep the smaller box" at 0.9. Prints the boxes and the timings.
 */
public class V3Onnx {

    static final String MODEL = "C:\\Users\\DELL\\SikuliX_VNC_optimized\\.claude\\A NETTOYER\\icon_detect_v3\\model.onnx";
    static final String[] OUTPUTS = {"logits_8", "dist_8", "logits_16", "dist_16", "logits_32", "dist_32"};
    static final int[] STRIDES = {8, 16, 32};
    static final int SIZE = 640;

    public static void main(String[] args) {
        String imagePath = args[0];
        float conf = args.length > 1 ? Float.parseFloat(args[1]) : 0.05f;
        String model = args.length > 2 ? args[2] : MODEL;

        long t0 = System.nanoTime();
        nu.pattern.OpenCV.loadLocally();
        System.out.printf("opencv       %d ms  %s  %s/%s%n", ms(t0), Core.VERSION,
                System.getProperty("os.name"), System.getProperty("os.arch"));

        t0 = System.nanoTime();
        Net net = Dnn.readNetFromONNX(model);
        System.out.printf("dnn load     %d ms%n", ms(t0));

        Mat bgr = Imgcodecs.imread(imagePath);
        int w = bgr.cols(), h = bgr.rows();
        double scale = Math.min((double) SIZE / w, (double) SIZE / h);
        int rw = (int) (w * scale), rh = (int) (h * scale);
        int padLeft = (SIZE - rw) / 2, padTop = (SIZE - rh) / 2;
        Mat resized = new Mat();
        Imgproc.resize(bgr, resized, new Size(rw, rh), 0, 0, Imgproc.INTER_LANCZOS4);
        Mat padded = new Mat(SIZE, SIZE, CvType.CV_8UC3, new Scalar(114, 114, 114));
        resized.copyTo(padded.submat(padTop, padTop + rh, padLeft, padLeft + rw));
        Mat blob = Dnn.blobFromImage(padded, 1.0 / 255.0, new Size(SIZE, SIZE), new Scalar(0, 0, 0), true, false);
        System.out.printf("image        %dx%d  conf=%.2f  blob %s%n", w, h, conf, Arrays.toString(blobShape(blob)));

        List<Mat> outs = new ArrayList<>();
        for (int run = 1; run <= 2; run++) {
            t0 = System.nanoTime();
            net.setInput(blob);
            outs = new ArrayList<>();
            net.forward(outs, Arrays.asList(OUTPUTS));
            System.out.printf("dnn forward %d %d ms%n", run, ms(t0));
        }

        t0 = System.nanoTime();
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        for (int level = 0; level < 3; level++) {
            Mat logits = outs.get(2 * level), dist = outs.get(2 * level + 1);
            int gh = logits.size(2), gw = logits.size(3), stride = STRIDES[level];
            float[] l = new float[gh * gw];
            float[] d = new float[4 * gh * gw];
            logits.get(new int[]{0, 0, 0, 0}, l);
            dist.get(new int[]{0, 0, 0, 0}, d);
            int cells = gh * gw;
            for (int y = 0; y < gh; y++) {
                for (int x = 0; x < gw; x++) {
                    int i = y * gw + x;
                    float s = (float) (1.0 / (1.0 + Math.exp(-l[i])));
                    if (s <= conf) continue;
                    double ax = (x + 0.5) * stride, ay = (y + 0.5) * stride;
                    double x1 = ax - d[i] * stride, y1 = ay - d[cells + i] * stride;
                    double x2 = ax + d[2 * cells + i] * stride, y2 = ay + d[3 * cells + i] * stride;
                    boxes.add(new Rect2d(x1, y1, x2 - x1, y2 - y1));
                    scores.add(s);
                }
            }
        }
        MatOfInt keep = new MatOfInt();
        float[] sc = new float[scores.size()];
        for (int i = 0; i < sc.length; i++) sc[i] = scores.get(i);
        Dnn.NMSBoxes(new MatOfRect2d(boxes.toArray(new Rect2d[0])), new MatOfFloat(sc), conf, 0.7f, keep);
        int[] kept = keep.toArray();

        List<double[]> finals = new ArrayList<>();   // x1 y1 x2 y2 score, image coordinates
        for (int i : kept) {
            Rect2d b = boxes.get(i);
            double x1 = clamp((b.x - padLeft) / scale, 0, w), y1 = clamp((b.y - padTop) / scale, 0, h);
            double x2 = clamp((b.x + b.width - padLeft) / scale, 0, w), y2 = clamp((b.y + b.height - padTop) / scale, 0, h);
            finals.add(new double[]{x1, y1, x2, y2, scores.get(i)});
        }
        List<double[]> smaller = keepSmaller(finals, 0.9);
        System.out.printf("decode       %d ms  %d candidates -> %d after NMS -> %d after overlap 0.9%n",
                ms(t0), boxes.size(), finals.size(), smaller.size());

        smaller.sort((a, b) -> Double.compare(b[4], a[4]));
        int rank = 0;
        for (double[] b : smaller) {
            System.out.printf("  #%d conf=%.3f  x=%.0f y=%.0f w=%.0f h=%.0f%n", rank++, b[4], b[0], b[1], b[2] - b[0], b[3] - b[1]);
        }
    }

    /** OmniParser "remove overlap": of two boxes overlapping above the threshold, the larger goes. */
    static List<double[]> keepSmaller(List<double[]> boxes, double threshold) {
        List<double[]> kept = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            double[] a = boxes.get(i);
            boolean drop = false;
            for (int j = 0; j < boxes.size() && !drop; j++) {
                if (j == i) continue;
                double[] b = boxes.get(j);
                drop = overlap(a, b) > threshold && area(a) > area(b);
            }
            if (!drop) kept.add(a);
        }
        return kept;
    }

    static double area(double[] b) {
        return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    }

    static double overlap(double[] a, double[] b) {
        double x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
        double x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
        double inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        double union = area(a) + area(b) - inter + 1e-6;
        return Math.max(inter / union, Math.max(inter / (area(a) + 1e-6), inter / (area(b) + 1e-6)));
    }

    static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    static int[] blobShape(Mat m) {
        int[] s = new int[m.dims()];
        for (int i = 0; i < s.length; i++) s[i] = m.size(i);
        return s;
    }

    static long ms(long t0) {
        return (System.nanoTime() - t0) / 1_000_000;
    }
}
