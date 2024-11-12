import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as scipy_zoom
from Resize_No_Vectorization import Resize  # Assuming Resize class from external module

# Hardcoded Java-generated downscaled and reverted images for comparison
downscaled_0_5_java_square_LS_3_3_3 = np.array([
    [0.0018689470814154346, -0.008251404860385576, -0.04985014714990277, -0.03066989849060565, 0.0013938326549015785],
    [-0.008251404860385763, 0.036429967893169046, 0.22008849291341548, 0.1354076592052344, -0.006153773778607437],
    [-0.04985014714990389, 0.2200884929134171, 1.3296455504694955, 0.8180536346012451, -0.03717749081293505],
    [-0.030669898490606213, 0.13540765920523523, 0.8180536346012426, 0.5033008600284541, -0.022873149520289068],
    [0.0013938326549015941, -0.006153773778607329, -0.037177490812933506, -0.022873149520288183, 0.0010394994535632193]
])

reverted_0_5_java_square_LS_3_3_3 = np.array([
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

downscaled_0_5_java_square_LS_1_1_1 = np.array([
    [0.006151480199923101, -0.012302960399846182, -0.09419454056132263, -0.061899269511726245, 0.00845828527489427],
    [-0.012302960399846198, 0.024605920799692396, 0.1883890811226452, 0.12379853902345257, -0.016916570549788554],
    [-0.0941945405613226, 0.18838908112264505, 1.4423539023452514, 0.947832564398308, -0.1295174932718186],
    [-0.061899269511726224, 0.12379853902345248, 0.9478325643983084, 0.622861399461745, -0.08511149557862364],
    [0.008458285274894286, -0.016916570549788523, -0.1295174932718186, -0.08511149557862356, 0.011630142252979614]
])

reverted_0_5_java_square_LS_1_1_1 = np.array([
    [0.0061514801955339276, -0.0030757400992221192, -0.012302960391831071, -0.05324875047573376, -0.09419454047476042, -0.07804690520636601, -0.061899268751859485, -0.026720494862330697, 0.008458295547883918, -0.026720530459464452],
    [-0.0030757400992220715, 0.0015378700503386135, 0.006151480198825903, 0.026624375250462698, 0.047097270259662136, 0.039023452621644875, 0.030949634390572044, 0.013360247437485854, -0.004229147775942612, 0.013360265236052804],
    [-0.012302960391831144, 0.0061514801988258394, 0.024605920785188546, 0.1064975009580736, 0.18838908096120768, 0.1560938104224148, 0.12379853751140003, 0.053440989727977124, -0.016916591096816872, 0.05344106092224331],
    [-0.0532487504757338, 0.026624375250463118, 0.10649750095807335, 0.4609344965924599, 0.8153714914921407, 0.6755935236131048, 0.5358155454667874, 0.23129928379601553, -0.07321712088194109, 0.23129959193370198],
    [-0.0941945404747611, 0.0470972702596626, 0.18838908096120655, 0.8153714914921434, 1.4423539007234187, 1.195093235726933, 0.9478325525681137, 0.4091575774953761, -0.12951765055036116, 0.40915812257648065],
    [-0.07804690520636626, 0.03902345262164537, 0.15609381042241488, 0.6755935236131031, 1.195093235726931, 0.9902201126671685, 0.7853469745586089, 0.33901627954546415, -0.10731462507387966, 0.33901673118409614],
    [-0.06189926875186017, 0.030949634390572398, 0.12379853751140012, 0.5358155454667877, 0.9478325525681169, 0.7853469745586079, 0.6228613846138586, 0.26887497644338093, -0.08511159796649252, 0.2688753346395321],
    [-0.026720494862330812, 0.013360247437486171, 0.053440989727977194, 0.23129928379601525, 0.4091575774953742, 0.3390162795454647, 0.26887497644338176, 0.11606716155994631, -0.03674072508522311, 0.11606731618501084],
    [0.008458295547883894, -0.004229147775942776, -0.016916591096817438, -0.07321712088194123, -0.12951765055035866, -0.10731462507387982, -0.0851115979664932, -0.036740725085223705, 0.0116301705120165, -0.036740774031340545],
    [-0.02672053045946421, 0.013360265236052908, 0.05344106092224384, 0.23129959193370056, 0.40915812257647916, 0.33901673118409564, 0.2688753346395318, 0.11606731618501057, -0.03674077403134107, 0.1160674708102806]
])

downscaled_0_5_java_square_Oblique_0_1_1 = np.array([
    [0.00237473057492863, -0.007124191724785899, -0.057091995031688554, -0.04017349730698399, 0.005745734457166682],
    [-0.007124191724785898, 0.02137257517435772, 0.17127598509506584, 0.1205204919209521, -0.017237203371500063],
    [-0.0570919950316885, 0.17127598509506567, 1.372575032768215, 0.9658296115233252, -0.1381358569874059],
    [-0.040173497306983975, 0.12052049192095204, 0.9658296115233252, 0.6796181019073095, -0.09720102574102238],
    [0.005745734457166682, -0.017237203371500063, -0.13813585698740585, -0.09720102574102243, 0.013901983155821643]
])

reverted_0_5_java_square_Oblique_0_1_1 = np.array([
    [0.002374730662670633, -0.002374730626326804, -0.007124191855106919, -0.03210809397163005, -0.0570919960863553, -0.04863274706789715, -0.040173498048520155, -0.017213881746601774, 0.005745734584774475, -0.017213881868000734],
    [-0.002374730626326799, 0.0023747305899829625, 0.007124191746075418, 0.03210809348023489, 0.05709199521259638, 0.04863274632360189, 0.04017349743368869, 0.017213881483153576, -0.005745734496839451, 0.01721388160455255],
    [-0.007124191855106922, 0.00712419174607541, 0.021372575166605758, 0.09632428011791752, 0.1712759850638347, 0.14589823848189484, 0.12052049189719695, 0.05164164427640752, -0.017237203432755782, 0.05164164464060419],
    [-0.03210809397162994, 0.03210809348023471, 0.09632428011791747, 0.43412489453925407, 0.7719255089362791, 0.6575502803330462, 0.5431750517173841, 0.23274426082275865, -0.07768653047015936, 0.2327442624641612],
    [-0.057091996086354804, 0.05709199521259587, 0.17127598506383457, 0.7719255089362784, 1.3725750327654929, 1.1692023221473729, 0.9658296115071527, 0.41384687735607584, -0.13813585750321233, 0.41384688027468397],
    [-0.048632747067897115, 0.04863274632360191, 0.14589823848189484, 0.6575502803330461, 1.1692023221473737, 0.9959630894352507, 0.8227238567043007, 0.35252770775179426, -0.11766844180398894, 0.352527710237956],
    [-0.04017349804852011, 0.04017349743368862, 0.12052049189719698, 0.5431750517173842, 0.9658296115071524, 0.822723856704301, 0.6796181018858971, 0.2912085381408489, -0.09720102610254121, 0.29120854019456416],
    [-0.017213881746601756, 0.017213881483153542, 0.051641644276407474, 0.23274426082275865, 0.41384687735607567, 0.3525277077517944, 0.29120853814084885, 0.12477950844865505, -0.04164952145707245, 0.12477950932864841],
    [0.005745734584774481, -0.0057457344968394565, -0.01723720343275575, -0.07768653047015933, -0.13813585750321208, -0.1176684418039889, -0.09720102610254133, -0.041649521457072416, 0.013901983259670646, -0.041649521750800986],
    [-0.017213881868000706, 0.017213881604552506, 0.05164164464060429, 0.232744262464161, 0.4138468802746841, 0.35252771023795654, 0.2912085401945637, 0.12477950932864847, -0.04164952175080101, 0.12477951020864177]
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
    downscaled_diff = downscaled_img - downscaled_0_5_java_square_LS_3_3_3
    reverted_diff = reverted_img - reverted_0_5_java_square_LS_3_3_3

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
    
    axs[1, 1].imshow(downscaled_0_5_java_square_LS_3_3_3, cmap='gray')
    axs[1, 1].set_title("Downscaled Image (Java)")

    axs[1, 2].imshow(downscaled_scipy, cmap='gray')
    axs[1, 2].set_title("Downscaled Image (SciPy)")

    # Reverted
    axs[2, 0].imshow(reverted_img, cmap='gray')
    axs[2, 0].set_title("Reverted Image (Python)")
    
    axs[2, 1].imshow(reverted_0_5_java_square_LS_3_3_3, cmap='gray')
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