# Generates a 3x10 grid of MNIST images at increasing degrees of noise
# Example usage
# python util/overlay.py --noise=gaussian --inc=0.5

from torchquantum.dataset import NoisyMNISTDataset
import matplotlib.pyplot as plt
import argparse
import os

FIG_SIZE = (17,7.5)
IMAGE_FOLDER = 'images/'
DEFAULT_INCREMENTS = {
        'gaussian': 0.5,
        'saltandpepper': 0.1,
        'poisson': 0.1,
        'speckle': 0.5,
    }

AXIS_TITLES = {
        'gaussian': 'Standard Deviation of Gaussian Noise',
        'saltandpepper': 'Percentage of Pixels of Salt and Pepper Noise',
        'poisson': 'Strength of Poisson Noise',
        'speckle': 'Standard Deviation of Speckle Noise'
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inc", type=float, help="Increment of std_dev variable"
    )
    parser.add_argument(
        "--figname", type=str, default="overlay", help="Name of the file, without file extension"
    )
    parser.add_argument(
        "--noise", type=str.lower, default="gaussian", help="Type of noise"
    )

    args = parser.parse_args()
    
    if args.noise not in DEFAULT_INCREMENTS:
        raise InvalidArgumentError(f"--noise must be one of {DEFAULT_INCREMENTS.keys()}")
    if args.inc is None:
        args.inc = DEFAULT_INCREMENTS[args.noise]
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER, exist_ok=True)

    ax_title = AXIS_TITLES[args.noise]

    images = []
    digits = [1,5,9]
    for i in range(10):
        mnist = NoisyMNISTDataset(
            root="mnist_data",
            split="train",
            noise=args.noise,
            train_valid_split_ratio=[0.9, 0.1],
            center_crop=28,
            resize=28,
            resize_mode="bilinear",
            binarize=False,
            binarize_threshold=0.1307,
            digits_of_interest=tuple(range(10)),
            n_test_samples=100,
            n_valid_samples=1000,
            fashion=False,
            n_train_samples=10000,
            std_dev=args.inc*i,
        )
        
        
        for digit in digits:
            item = None
            ind = 0
            for item in mnist:
                if item['digit'] == digit: break
            image_tensor = item['image'].squeeze()
            images.append(image_tensor)


    # Define the dimensions of the grid
    num_rows = 3
    num_cols = 10
    ind = 0
    # Create a figure and axis
    fig, axes = plt.subplots(num_rows, num_cols, figsize=FIG_SIZE)

    # Iterate over the axes and images
    for j in range(num_cols):
        for i in range(num_rows):
            img = images[ind]
            
            # Display the image
            axes[i, j].imshow(img, cmap='gray')

            axes[i,j].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

            axes[i,j].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,
            labelleft=False)
            
            #axes[i, j].axis('off')

            ind += 1

    # Adjust layout
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.01, right=0.99, top=0.875, bottom=0.2)

    for j in range(num_cols):
        axes[num_rows-1, j].set_xlabel(str(round(args.inc*j,3)), fontsize=30, labelpad=10)

    plt.figtext(.5,.91,"MNIST Digits with Increasing Degrees of Noise Added",fontsize=40,ha='center')
    plt.figtext(.5,.04,ax_title,fontsize=40,ha='center')


    plt.savefig(IMAGE_FOLDER + args.figname+'.eps', format='eps')
    plt.savefig(IMAGE_FOLDER + args.figname+'.png')

if __name__ == '__main__':
    main()