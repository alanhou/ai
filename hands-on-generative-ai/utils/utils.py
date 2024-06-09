__all__ = ['show_image', 'subplots', 'get_grid', 'show_images', 'init_ddpm', 'load_image', 'image_grid',
           'plot_scheduler',
           'plot_noise_and_denoise', 'spectrogram_from_image', 'waveform_from_spectrogram',
           'wav_bytes_from_spectrogram_image', 'measure_latency_and_memory_use', 'device', 'load_custom_dataset',
           'corrupt', 'get_stable_diffusion_pipeline', 'get_autoencoder', 'make_image_grid', 'color_loss','download_image']

import gc
import math
import typing
from io import BytesIO
from itertools import zip_longest

import fastcore.all as fc
import numpy as np
import requests
import torchaudio
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch.nn import init
from torchvision.utils import make_grid
from transformers import set_seed

import torch
from PIL import Image
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL, StableDiffusionPipeline
from torchvision.transforms import transforms


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ("cpu", "permute", "detach")):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
        nrows: int = 1,  # Number of rows in returned axes grid
        ncols: int = 1,  # Number of columns in returned axes grid
        figsize: tuple = None,  # Width, height in inches of the returned figure
        imsize: int = 3,  # Size (in inches) of images that will be displayed in the returned figure
        suptitle: str = None,  # Title to be set to returned figure
        **kwargs,
):  # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None:
        figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
        n: int,  # Number of axes
        nrows: int = None,  # Number of rows, defaulting to `int(math.sqrt(n))`
        ncols: int = None,  # Number of columns, defaulting to `ceil(n/rows)`
        title: str = None,  # If passed, title set to the figure
        weight: str = "bold",  # Title font weight
        size: int = 14,  # Title font size
        **kwargs,
):  # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows:
        ncols = ncols or int(np.floor(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(
        ims: list,  # Images to show
        nrows: typing.Union[int, None] = None,  # Number of rows in grid
        ncols: typing.Union[
            int, None
        ] = None,  # Number of columns in grid (auto-calculated if None)
        titles: typing.Union[
            list, None
        ] = None,  # Optional list of titles for each image
        **kwargs,
):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs):
        show_image(im, ax=ax, title=t)


def init_ddpm(model):
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers):
                init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()


def load_image(url, size=None, return_tensor=False):
    if url.startswith("http"):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(url)
    if size is not None:
        img = img.resize(size)
    if return_tensor:
        return TF.to_tensor(img)
    return img


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def plot_scheduler(scheduler, ax=None, plot_both=True, label=None):
    if ax is None:
        fig, (ax) = plt.subplots(1, 1)
    # Check if SimpleScheduler
    if not hasattr(scheduler, "alphas_cumprod"):
        ax.plot(
            torch.linspace(1, 0, scheduler.num_train_timesteps),
            label=r"${\sqrt{\bar{\alpha}_t}}$ equivalent",
        )
        if plot_both:
            ax.plot(
                torch.linspace(0, 1, scheduler.num_train_timesteps),
                label=r"$\sqrt{(1 - \bar{\alpha}_t)}$ equivalent",
            )
        ax.legend()
        return
    if label is None:
        label = r"${\sqrt{\bar{\alpha}_t}}$"
    ax.plot(scheduler.alphas_cumprod.cpu() ** 0.5, label=label)
    if plot_both:
        ax.plot(
            (1 - scheduler.alphas_cumprod.cpu()) ** 0.5,
            label=r"$\sqrt{(1 - \bar{\alpha}_t)}$",
        )
    ax.legend(fontsize="x-large")
    plt.plot()


def plot_noise_and_denoise(scheduler_output, step):
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    prev_prev_sample = scheduler_output.prev_sample
    grid = make_grid(prev_prev_sample, nrow=4).permute(1, 2, 0)
    axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
    axs[0].set_title(f"Current x (step {step})")
    plt.axis("off")

    pred_x0 = scheduler_output.pred_original_sample
    grid = make_grid(pred_x0, nrow=4).permute(1, 2, 0)
    axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
    axs[1].set_title(f"Predicted denoised images (step {step})")
    plt.axis("off")
    plt.show()


"""
Simplified from riffusion codebase.
"""


def spectrogram_from_image(image, max_volume, power_for_image) -> np.ndarray:
    # Convert to a numpy array of floats
    data = np.array(image).astype(np.float32)

    # Flip vertically and take a single channel
    data = data[::-1, :, 0]

    # Invert
    data = 255 - data

    # Rescale to max volume
    data = data * max_volume / 255

    # Reverse the power curve
    data = np.power(data, 1 / power_for_image)

    return data


def waveform_from_spectrogram(
        Sxx: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_samples: int,
        sample_rate: int,
        n_mels: int,
        max_mel_iters: int,
        num_griffin_lim_iters: int,
        device: str = "cpu",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.
    This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
    to approximate the phase.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    mel_inv_scaler = torchaudio.transforms.InverseMelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0,
        f_max=10000,
        n_stft=n_fft // 2 + 1,
        norm=None,
        mel_scale="htk",
        max_iter=max_mel_iters,
    ).to(device)

    Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform


def wav_bytes_from_spectrogram_image(image, device="cpu"):
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """

    max_volume = 50
    power_for_image = 0.25
    Sxx = spectrogram_from_image(
        image, max_volume=max_volume, power_for_image=power_for_image
    )

    sample_rate = 44100  # [Hz]
    clip_duration_ms = 5000  # [ms]

    bins_per_image = 512
    n_mels = 512

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = (
            int(image.width / float(bins_per_image) * clip_duration_ms)
            * sample_rate
    )
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        max_mel_iters=200,
        num_griffin_lim_iters=32,
        device=device,
    )

    wav_bytes = BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    return wav_bytes


def measure_latency_and_memory_use(
        pipeline, inputs, model_name, device, nb_loops=50
):
    # Define Events that measure start and end of the generate pass
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Reset cuda memory stats and empty cache
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()  # Needed due to Ubuntu

    # Get the start time
    start_event.record()

    # Perform generation
    for _ in range(nb_loops):
        set_seed(0)
        _ = pipeline(inputs)

    # Get end time
    end_event.record()
    torch.cuda.synchronize()

    # Measure memory footprint and elapsed time
    max_memory = torch.cuda.max_memory_allocated(device)
    elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

    print(f"{model_name} execution time: {elapsed_time / nb_loops} seconds")
    print(f"{model_name} max memory footprint: {max_memory * 1e-9} GB")


access_token = "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"  # Your access token of Huggingface


def access():
    return access_token


def load_custom_dataset(name):
    return load_dataset("huggan/" + name, split="train",
                        token=access_token)


def transform(examples):
    image_size = 64

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def train_dataloader(dataset):
    batch_size = 32
    dataset.set_transform(transform)
    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return train_data_loader


def scheduler():
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)


def unet_model():
    return UNet2DModel(
        in_channels=3,  # 3 channels for RGB images
        sample_size=64,  # Specify our input size
        block_out_channels=(64, 128, 256, 512),  # N channels per layer
        down_block_types=("DownBlock2D", "DownBlock2D",
                          "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D",
                        "UpBlock2D", "UpBlock2D"),
    )


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def corrupt(x, noise, amount):
    amount = amount.view(-1, 1, 1, 1)  # make sure it's broadcastable

    return x * (1 - amount) + noise * amount


def get_stable_diffusion_pipeline(device):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", vae=vae).to(device)
    return pipe


def get_autoencoder(device):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device)


def make_image_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def color_loss(images, target_color=(0.1, 0.9, 0.5)):
    """Given a target color (R, G, B) return a loss for how far away on average
    the images' pixels are from that color. Defaults to a light teal: (0.1, 0.9, 0.5)"""
    target = (
            torch.tensor(target_color).to(images.device) * 2 - 1
    )  # Map target color to (-1, 1)
    target = target[
             None, :, None, None
             ]  # Get shape right to work with the images (b, c, h, w)
    error = torch.abs(
        images - target
    ).mean()  # Mean absolute difference between the image pixels and the target color
    return error


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")
