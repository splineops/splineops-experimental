import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as scipy_zoom
from Resize_No_Vectorization import Resize  # Assuming Resize class from external module

# Hardcoded Java-generated downscaled and reverted images for comparison
downscaled_java = np.array([
    [0.0018689470814154346, -0.008251404860385576, -0.04985014714990277, -0.03066989849060565, 0.0013938326549015785],
    [-0.008251404860385763, 0.036429967893169046, 0.22008849291341548, 0.1354076592052344, -0.006153773778607437],
    [-0.04985014714990389, 0.2200884929134171, 1.3296455504694955, 0.8180536346012451, -0.03717749081293505],
    [-0.030669898490606213, 0.13540765920523523, 0.8180536346012426, 0.5033008600284541, -0.022873149520289068],
    [0.0013938326549015941, -0.006153773778607329, -0.037177490812933506, -0.022873149520288183, 0.0010394994535632193]
])

reverted_java = np.array([
    [0.0018781323400760897, 0.0010571984329069463, -0.008262973560162083, -0.031004085484624598, -0.04994430169807815, -0.048221066443239174, -0.03064737786618124, -0.00944189116442014, 0.0017383923557298085, -0.009896553700642022],
    [0.0010571984329074248, 5.950957249894561E-4, -0.004651217868177059, -0.01745216238958293, -0.028113587291633116, -0.02714358024147593, -0.017251372100620554, -0.005314829168172281, 9.785389639727809E-4, -0.005570758162376665],
    [-0.008262973560158452, -0.004651217868181094, 0.03635352557381595, 0.13640462556876864, 0.21973342112605143, 0.2121519280410835, 0.13483526564821757, 0.04154025538284788, -0.007648177801984145, 0.04354057476150174],
    [-0.031004085484616906, -0.017452162389586988, 0.13640462556877103, 0.5118134096450134, 0.8244772566221064, 0.7960302020605083, 0.5059249036758746, 0.15586612005549, -0.028697266988463928, 0.1633716593823823],
    [-0.04994430169808113, -0.02811358729163959, 0.21973342112606764, 0.8244772566220873, 1.3281456364312492, 1.2823204410339848, 0.8149914964688332, 0.2510838298527845, -0.04622826114622253, 0.2631744596742541],
    [-0.04822106644322524, -0.027143580241484618, 0.21215192804110306, 0.7960302020604646, 1.2823204410340732, 1.238076358788619, 0.7868717304218792, 0.24242064921323506, -0.044633240960376956, 0.25409411433617746],
    [-0.030647377866186362, -0.017251372100625433, 0.1348352656482193, 0.5059249036758733, 0.8149914964688023, 0.7868717304218127, 0.5001041460341442, 0.15407285211640753, -0.02836709973464651, 0.16149203885400695],
    [-0.00944189116442196, -0.005314829168163082, 0.04154025538285008, 0.15586612005550343, 0.2510838298528118, 0.24242064921333514, 0.15407285211635763, 0.04746700051885365, -0.008739379581333187, 0.049752714944766116],
    [0.001738392355726042, 9.785389639690011E-4, -0.007648177801995245, -0.02869726698848693, -0.04622826114624887, -0.0446332409604733, -0.028367099734603184, -0.008739379581321288, 0.001609049542452153, -0.00916021354522421],
    [-0.009896553700638518, -0.005570758162372193, 0.0435405747615197, 0.1633716593823974, 0.26317445967427133, 0.2540941143362428, 0.16149203885396476, 0.049752714944765124, -0.00916021354522903, 0.05214849510856498]
])

def resize_image_with_least_squares(zoom_y, zoom_x):
    # Create a 10x10 square image
    input_img = np.zeros((10, 10))
    input_img[3:7, 3:7] = 255.0  # White square in the center
    
    # Normalize the input image to the range [0, 1]
    input_img_normalized = input_img / 255.0

    input_img_normalized_copy = input_img_normalized.copy()

    # Define output shape for downscaled image based on zoom factors
    output_shape = (int(round(input_img.shape[0] * zoom_y)), int(round(input_img.shape[1] * zoom_x)))
    downscaled_img = np.zeros(output_shape, dtype=np.float64)

    # Initialize Resize instance
    resizer = Resize()
    
    # Set parameters for least-squares cubic interpolation
    interp_degree = 3  # Cubic interpolation
    synthe_degree = 3
    analy_degree = 3
    shift_y = 0.0  # No shift
    shift_x = 0.0
    inversable = False  # Non-inversible for this example

    # Step 1: Downscale to 5x5 using custom Resize class
    resizer.compute_zoom(
        input_img=input_img_normalized, 
        output_img=downscaled_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=zoom_y, 
        zoom_x=zoom_x, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    # Downscale with scipy.ndimage.zoom for comparison
    downscaled_scipy = scipy_zoom(input_img_normalized, (zoom_y, zoom_x), order=3)

    # Step 2: Resize back to 10x10 using custom Resize class
    reverted_shape = (10, 10)
    reverted_img = np.zeros(reverted_shape, dtype=np.float64)
    reverse_zoom_y = 1.0 / zoom_y
    reverse_zoom_x = 1.0 / zoom_x

    downscaled_img_copy = downscaled_img.copy()

    resizer.compute_zoom(
        input_img=downscaled_img, 
        output_img=reverted_img, 
        analy_degree=analy_degree, 
        synthe_degree=synthe_degree, 
        interp_degree=interp_degree, 
        zoom_y=reverse_zoom_y, 
        zoom_x=reverse_zoom_x, 
        shift_y=shift_y, 
        shift_x=shift_x, 
        inversable=inversable
    )

    # Revert using scipy.ndimage.zoom for comparison
    reverted_scipy = scipy_zoom(downscaled_scipy, (reverse_zoom_y, reverse_zoom_x), order=3)

    # Compute differences
    downscaled_diff = downscaled_img - downscaled_java
    reverted_diff = reverted_img - reverted_java

    print(downscaled_diff)
    print(reverted_diff)

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    # Original input
    axs[0, 0].imshow(input_img_normalized_copy, cmap='gray')
    axs[0, 0].set_title("Initial Square Image (Python)")

    axs[0, 1].imshow(input_img_normalized_copy, cmap='gray')
    axs[0, 1].set_title("Initial Square Image (Java)")

    axs[0, 2].imshow(input_img_normalized_copy, cmap='gray')
    axs[0, 2].set_title("Initial Square Image (SciPy)")

    # Downscaled
    axs[1, 0].imshow(downscaled_img_copy, cmap='gray')
    axs[1, 0].set_title("Downscaled Image (Python)")
    
    axs[1, 1].imshow(downscaled_java, cmap='gray')
    axs[1, 1].set_title("Downscaled Image (Java)")

    axs[1, 2].imshow(downscaled_scipy, cmap='gray')
    axs[1, 2].set_title("Downscaled Image (SciPy)")

    # Reverted
    axs[2, 0].imshow(reverted_img, cmap='gray')
    axs[2, 0].set_title("Reverted Image (Python)")
    
    axs[2, 1].imshow(reverted_java, cmap='gray')
    axs[2, 1].set_title("Reverted Image (Java)")

    axs[2, 2].imshow(reverted_scipy, cmap='gray')
    axs[2, 2].set_title("Reverted Image (SciPy)")

    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with 10x10 square image:
resize_image_with_least_squares(
    zoom_y=0.5,  # Vertical zoom factor
    zoom_x=0.5   # Horizontal zoom factor
)
