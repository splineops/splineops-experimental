public class comparison_least_squares {
    public static void main(String[] args) {
        // Create a 10x10 black image with a 5x5 white square in the middle
        ImageAccess input = new ImageAccess(10, 10);
        ImageAccess downscaled = new ImageAccess(5, 5);  // Intermediate 5x5 image
        ImageAccess reverted = new ImageAccess(10, 10);  // Resized-back 10x10 image

        // Fill the input image with a white square
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                if (x >= 3 && x < 7 && y >= 3 && y < 7) {
                    input.putPixel(x, y, 1.0);  // White pixel, normalized
                } else {
                    input.putPixel(x, y, 0.0);  // Black pixel
                }
            }
        }

        // Initialize the Resize instance and set parameters for least-squares cubic interpolation
        Resize resizer = new Resize();
        int degree = 3;  // Cubic interpolation degree
        double zoomFactor = 0.5;

        // Step 1: Downscale the image to 5x5
        resizer.computeZoom(input, downscaled, degree, degree, degree, zoomFactor, zoomFactor, 0, 0, false);

        // Step 2: Upscale back to 10x10
        double reverseZoomFactor = 1.0 / zoomFactor;
        resizer.computeZoom(downscaled, reverted, degree, degree, degree, reverseZoomFactor, reverseZoomFactor, 0, 0, false);

        // Print input, downscaled, and reverted images
        System.out.println("Original Input Image (10x10):");
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                System.out.print(input.getPixel(x, y) + " ");
            }
            System.out.println();
        }

        System.out.println("\nDownscaled Image (5x5):");
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                System.out.print(downscaled.getPixel(x, y) + " ");
            }
            System.out.println();
        }

        System.out.println("\nReverted Image (10x10):");
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                System.out.print(reverted.getPixel(x, y) + " ");
            }
            System.out.println();
        }
    }
}
