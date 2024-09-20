public class ImageAccess {
    public static final int FLOAT = 1;  // Added this constant to represent float type

    private double[][] pixels;
    private int width;
    private int height;

    // Constructor
    public ImageAccess(int width, int height) {
        this.width = width;
        this.height = height;
        this.pixels = new double[height][width];
    }

    // Constructor with data type (currently ignored as we're using double for all data)
    public ImageAccess(int width, int height, int dataType) {
        this(width, height);  // Call the main constructor
        if (dataType != FLOAT) {
            throw new IllegalArgumentException("Unsupported data type");
        }
    }

    // Get the width of the image
    public int getWidth() {
        return width;
    }

    // Get the height of the image
    public int getHeight() {
        return height;
    }

    // Get pixel value at (x, y)
    public double getPixel(int x, int y) {
        return pixels[y][x];
    }

    // Set pixel value at (x, y)
    public void putPixel(int x, int y, double value) {
        pixels[y][x] = value;
    }

    // Get a column of pixels
    public void getColumn(int col, double[] column) {
        for (int row = 0; row < height; row++) {
            column[row] = pixels[row][col];
        }
    }

    // Put a column of pixels
    public void putColumn(int col, double[] column) {
        for (int row = 0; row < height; row++) {
            pixels[row][col] = column[row];
        }
    }

    // Get a row of pixels
    public void getRow(int row, double[] rowData) {
        System.arraycopy(pixels[row], 0, rowData, 0, width);
    }

    // Put a row of pixels
    public void putRow(int row, double[] rowData) {
        System.arraycopy(rowData, 0, pixels[row], 0, width);
    }
}
