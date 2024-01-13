#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import skimage as SK


def ShowTime(
    Image,
    figsize=(8, 6),
    bins=32,
    Im_title="Image",
    With_Hist=False,
    His_title="Histogram",
    Save=False,
    Save_Name="Day06",
    format="png",
    cmap="gray",
    colorbar=True,
):
    """
    Awesome Image shower with histogram subplot
    With capability of saving Output as any format like {'png','pdf','eps',...}
    """
    if With_Hist:
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)
        fig.subplots_adjust(wspace=0.2)
        a = ax[0].imshow(Image, cmap=cmap)
        Loc = make_axes_locatable(ax[0])
        Cax = Loc.append_axes("bottom", size="5%", pad="2%")
        ax[0].set_title(Im_title)
        ax[0].axis("off")
        if colorbar:
            fig.colorbar(a, cax=Cax, shrink=0.9, orientation="horizontal")

        # Hist,Bins = np.histogram(Image.ravel(),bins=range(bins))
        Hist, Bins = SK.exposure.histogram(Image, nbins=bins)
        CDF, _ = SK.exposure.cumulative_distribution(Image, nbins=bins)

        ax[1].bar(Bins, Hist, ec="black", fc="blue", width=2.0)
        Ex = ax[1].twinx()
        Ex.set_ylim(0, 1.2)
        Ex.plot(Bins, CDF, "r", lw=2.0)
        ax[1].set_title(His_title)
        plt.show()
        if Save:
            fig.savefig(Save_Name + "." + format, format=format)
    else:
        a = plt.imshow(Image, cmap=cmap)
        plt.axis("off")
        plt.title(Im_title)
        if colorbar:
            plt.colorbar(a, shrink=0.7, orientation="horizontal")
        plt.show()
        if Save:
            plt.savefig(Save_Name + "." + format, format=format)


def ShowTimeMul(
    Images,
    figsize=(8, 6),
    bins=32,
    Im_title=None,
    With_Hist=False,
    His_Title=None,
    colorbar=True,
    cmaps="gray",
):
    """
    Multi Image show time plotter.
    """
    if len(cmaps) == 1:
        cmaps = cmaps[0]
    n = len(Images)
    if With_Hist:
        fig, ax = plt.subplots(nrows=n, ncols=2, figsize=figsize)
        fig.subplots_adjust(wspace=0.2)

        for i in range(n):
            a = ax[i, 0].imshow(Images[i], cmap=cmaps[i])
            ax[i, 0].axis("off")
            ax[i, 0].set_title(Im_title[i])
            Loc = make_axes_locatable(ax[i, 0])
            Cax = Loc.append_axes("bottom", size="5%", pad="2%")

            if colorbar:
                fig.colorbar(a, cax=Cax, shrink=0.9, orientation="horizontal")

            Hist, Bins = SK.exposure.histogram(Images[i], nbins=bins)
            CDF, _ = SK.exposure.cumulative_distribution(Images[i], nbins=bins)
            ax[i, 1].bar(Bins, Hist, ec="black", fc="blue")
            Ex = ax[i, 1].twinx()
            Ex.plot(Bins, CDF, "r", lw=2.5)
            ax[i, 1].set_title(His_Title[i])
        fig.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(nrows=1, ncols=n, figsize=figsize)

        for i in range(n):
            a = ax[i].imshow(Images[i], cmap=cmaps[i])
            ax[i].axis("off")
            ax[i].set_title(Im_title[i])
            if colorbar:
                fig.colorbar(a, shrink=0.8, orientation="horizontal")
        fig.tight_layout()
        plt.show()


def Zero_Padding(Image, height, width):
    """
    Simple zero padding function.
    """
    M, N = Image.shape
    result = np.zeros((M + 2 * height, N + 2 * width))
    result[height:-height, width:-width] = Image
    return result


def Zero_Unpadding(Image, height, width):
    """
    Simple zero unpadding function.
    """
    result = Image[height:-height, width:-width]
    return result


def FastConv2D(Image, Kernel):
    """
    Faster Convolution of 2D image with kernel.
    """
    M, N = Image.shape
    K1, K2 = Kernel.shape
    result = np.zeros_like(Image)
    Kernel = np.fliplr(np.flipud(Kernel))
    Padded_Image = Zero_Padding(Image, K1 // 2, K2 // 2)

    for y in range(M):
        for x in range(N):
            result[y, x] = np.sum(Padded_Image[y : y + K1, x : x + K2] * Kernel)

    return result


def Convolution2D(Image, Kernel):
    """
    Simple convolution of Image with kernel.
    """
    M, N = Image.shape
    K1, K2 = Kernel.shape
    result = np.zeros_like(Image)
    Kernel = np.fliplr(np.flipud(Kernel))

    for y in range(M):
        for x in range(N):
            for i in range(K1):
                for j in range(K2):
                    if (
                        y + i - K1 // 2 >= 0
                        and y + i - K1 // 2 < M
                        and x + j - K2 // 2 >= 0
                        and x + j - K2 // 2 < N
                    ):
                        result[y, x] += (
                            Kernel[i, j] * Image[y + i - K1 // 2, x + j - K2 // 2]
                        )

    return result


def Grad(Image):
    """
    Simple Gradient magnitude and angles calculator
    """
    Dx = SK.filters.sobel_h(Image)
    Dy = SK.filters.sobel_v(Image)
    M = np.sqrt(Dx**2+Dy**2)
    th = np.rad2deg(np.arctan2(Dy,Dx))
    th = (th + 360)%360
    return M,th

def NLS(Magnitude,Theta):
    """The Simple Non local supression function"""
    M,N = Magnitude.shape
    result = np.zeros_like(Magnitude)

    ## Splitting the angles to 45 degree lines
    Theta = (np.floor((Theta+22.5)/45)*45)%360
    for i in range(1,M-1):
        for j in range(1,N-1):
            alpha = Theta[i,j]

            if alpha == 0.0 or alpha == 180.0:
                Indexs = [Magnitude[i,j-1],Magnitude[i,j+1]]
            elif alpha == 45.0 or alpha == 225.0:
                Indexs = [Magnitude[i+1,j+1],Magnitude[i-1,j-1]]
            elif alpha == 90.0 or alpha == 270.0:
                Indexs = [Magnitude[i-1,j],Magnitude[i+1,j]]
            elif alpha == 135.0 or alpha == 315.0:
                Indexs = [Magnitude[i+1,j],Magnitude[i-1,j+1]]
            else:
                raise ValueError(f"The alpha = {alpha} is not in [0,45,90,135,180,225,270,315].")
            
            if Magnitude[i,j] >= np.max(Indexs):
                result[i,j] = Magnitude[i,j]
            else:
                result[i,j] = 0.0
            
    return result


def gauss(size,sigma):
    x = np.linspace(-(size)/2,(size)/2,size)
    tmp =1/(np.sqrt(2*np.pi*sigma)) * np.exp(-0.5*x**2/sigma**2)
    res = np.outer(tmp,tmp)
    return res/res.sum()
