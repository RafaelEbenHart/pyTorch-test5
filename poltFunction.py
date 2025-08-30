import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
from typing import Dict,List,Tuple


def plot_transformed_images(image_path: list, transform, n=3, seed=None):
        """
        selectes random image from a path of images
        and loads/trainsform then plots the orginial and trannsformed version
        """
        if seed:
            random.seed(seed)
        random_image_path = random.sample(image_path, k=n)
        for image_path in random_image_path:
            with Image.open(image_path) as f:
                fig,ax = plt.subplots(nrows=1, ncols=2)
                ax[0].imshow(f)
                ax[0].set_title(f"original\nShape: {f.size}")
                ax[0].axis(False)

                # Transform and plot target images
                transformed_image = transform(f).permute(1, 2, 0) # note : need to change shape for matplotlib to color_channel last
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"transformed\nShape: {transformed_image.shape}")
                ax[1].axis(False)

                fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


# visualize data(image)
def display_random_image(dataset: torch.utils.data.Dataset,
                          classes: list[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"for display, purpose, N shouldn't be larger than 10,setting to 10 and removing shape display")
    if seed:
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(10,5))
    for i, targ_samples in enumerate(random_samples_idx):
        targ_img, targ_label = dataset[targ_samples][0], dataset[targ_samples][1]
        targ_img_adjust = targ_img.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_img_adjust)
        plt.axis(False)
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title= title + f"\nshape: {targ_img_adjust.shape}"
        plt.title(title,fontsize=5)
    # plt.show()

## plot loss curves of model
def plot_loss_curves(results: Dict[str,List[float]]):
        """Plots training curves of a results dictionary"""
        # get the loss values of the results dictionary(train and test)
        train_loss = results["train_loss"]
        test_loss = results["test_loss"]
        # get the acc values of the results dictionary(train and test)
        train_acc = results["train_acc"]
        test_acc = results["test_acc"]
        # figure out how many epochs
        epochs = range(len(results["train_loss"]))
        # setup plot
        plt.figure(figsize=(10,8))

        # plot the loss
        plt.subplot(1, 2 ,1)
        plt.plot(epochs,train_loss,label="Train Loss")
        plt.plot(epochs,test_loss,label="Test Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # plot the accucary
        plt.subplot(1, 2, 2)
        plt.plot(epochs,train_acc,label="Train Acc")
        plt.plot(epochs,test_acc,label="Test Acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()