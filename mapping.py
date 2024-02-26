import multiprocessing
import os
import time
import numpy as np
from scipy import optimize
from numba import jit

from sklearn.neighbors import KNeighborsRegressor
import pickle

import torch


class custom_cache:
    def __init__(self, function):
        self.cache = {}
        self.function = function

    def __call__(self, *args, **kwargs):
        key = b""
        for arg in args:
            if isinstance(arg, np.ndarray):
                key += arg.data.tobytes()
            elif isinstance(arg, torch.Tensor):
                key += arg.numpy().data.tobytes()
            else:
                key += str(arg).encode("utf-8")

        for k, v in kwargs.items():
            key += str(k).encode("utf-8")
            if isinstance(v, np.ndarray):
                key += v.data.tobytes()
            elif isinstance(v, torch.Tensor):
                key += v.numpy().data.tobytes()
            else:
                key += str(v).encode("utf-8")

        if key in self.cache:
            return self.cache[key]

        value = self.function(*args, **kwargs)
        self.cache[key] = value
        return value


def solve_single_t1(signal: np.ndarray, times: np.ndarray) -> float:
    """
    Solves for T1 using the signal and times of a single voxel.
    The signals and times have shape (T,) where T is the number of timesteps.
    """
    params, _ = optimize.curve_fit(
        intensity_model_t1,
        times,
        signal,
        p0=[6, 3, 1000],
        maxfev=10000,
        bounds=((0, 0, 0), (np.inf, np.inf, np.inf)),
    )
    A, B, T1star = params

    return T1star * (B / A - 1)


@custom_cache
def compute_t1_mapping(times: np.ndarray, images: np.ndarray, KNN: bool = True, multi_coil: bool = False) -> np.ndarray:
    """
    Computes T1 mapping from the given times and images.
    Times has shape (T, S) where T is the number of timesteps and S is the number of slices.
    Images has shape (T, S, Ks, Kx, Ky) where Ks is the number of slices and Kx, Ky are the image dimensions.
    """
    if times.shape[0] != 9:
        times = times.T

    if images.shape[0] != 9:
        images = images.swapaxes(1, 0)

    images = np.abs(np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1)))

    if multi_coil:
        images = np.sqrt(np.sum(np.abs(images) ** 2, axis=-3))

    T, S = times.shape
    # Number of frequency encodings
    Ks, Kx, Ky = images.shape[1:]

    flat_images = images.reshape(T, Ks * Kx * Ky).T
    flat_times = np.tile(times[:, :, None, None], (1, 1, Kx, Ky)).reshape(T, S * Kx * Ky).T

    flat_images[:, :3] = -flat_images[:, :3]

    knn_exists = os.path.exists("knn.pkl")
    if not KNN or not knn_exists:
        with multiprocessing.Pool() as pool:
            T1 = pool.starmap(solve_single_t1_faster_solver, zip(flat_images, flat_times))

    if not knn_exists:
        knn = KNeighborsRegressor(n_neighbors=7, leaf_size=1, p=2, n_jobs=-1)
        reg = knn.fit(flat_images, T1)

        with open("knn.pkl", "wb") as f:
            pickle.dump(reg, f)

    if KNN:
        with open("knn.pkl", "rb") as f:
            reg = pickle.load(f)
        T1 = reg.predict(flat_images)

    T1 = np.array(T1).reshape(Ks, Kx, Ky)

    return T1


@custom_cache
def solve_single_t2(signal: np.ndarray, times: np.ndarray) -> float:
    """
    Solves for T2 using the signal and times of a single voxel.
    The signals and times have shape (T,) where T is the number of timesteps.
    """
    params, _ = optimize.curve_fit(
        intensity_model_t2,
        times,
        signal,
        p0=[1, 50],
        maxfev=10000,
        bounds=([0, 0], [np.inf, np.inf]),
    )
    return np.array(params[1])


def compute_t2_mapping(
    times: np.ndarray,
    images: np.ndarray,
    iterative: bool = False,
    multi_coil: bool = False,
) -> np.ndarray:
    """
    Computes T2 mapping from the given times and images.
    Times has shape (T, S) where T is the number of timesteps and S is the number of slices.
    Images has shape (T, S, Ks, Kx, Ky) where Ks is the number of slices and Kx, Ky are the image dimensions.

    If iterative is True, then the T2 mapping is computed iteratively, otherwise it is by solving a linearised problem with least squares.
    """
    if times.shape[0] != 3:
        times = times.T

    if times.shape[1] > 1 and (times[:, 1] == 0).all():
        times = times[:, 0:1]

    if images.shape[0] != 3:
        images = images.swapaxes(1, 0)

    images = np.abs(np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1)))

    if multi_coil:
        images = np.sqrt(np.sum(np.abs(images) ** 2, axis=-3))

    T, S = times.shape
    # Number of frequency encodings
    Ks, Kx, Ky = images.shape[1:]

    same_times = Ks != S and S == 1

    if iterative:
        # parallelise the code below
        flat_images = images.reshape(T, Ks * Kx * Ky).T
        if same_times:
            flat_times = np.tile(times[:, :, None, None], (1, Ks, Kx, Ky)).reshape(T, Ks * Kx * Ky).T
        else:
            flat_times = np.tile(times[:, :, None, None], (1, 1, Kx, Ky)).reshape(T, S * Kx * Ky).T

        with multiprocessing.Pool() as pool:
            T2 = pool.starmap(solve_single_t2, zip(flat_images, flat_times))

        T2 = np.array(T2).reshape(Ks, Kx, Ky)
    else:
        flat_images = images.reshape(T, Ks * Kx * Ky).T
        if same_times:
            flat_times = np.tile(times[:, :, None, None], (1, Ks, Kx, Ky)).reshape(T, Ks * Kx * Ky).T
        else:
            flat_times = np.tile(times[:, :, None, None], (1, 1, Kx, Ky)).reshape(T, S * Kx * Ky).T

        log_signal = np.log(flat_images)

        X = np.stack((flat_times, np.ones_like(flat_times)), axis=-1)
        y = log_signal[:, :, None]
        # alpha = np.linalg.inv(X.transpose(0, 2, 1) @ X) @ X.transpose(0, 2, 1) @ y
        alpha = np.linalg.pinv(X) @ y
        alpha = alpha.reshape(Ks, Kx, Ky, 2)

        minus_1_over_t2 = alpha[..., 0]
        T2 = -1 / minus_1_over_t2
        T2[T2 < 0] = 0
        T2[T2 > 250] = 250

    return T2


@jit(nopython=True)
def intensity_model_t1(t: float, A: float, B: float, T1star: float) -> float:
    """Model of signal intensity vs. time"""
    return A - B * np.exp(-t / T1star)


@jit(nopython=True)
def intensity_model_t2(t: float, A: float, T2: float) -> float:
    """Model of signal intensity vs. time"""
    return A * np.exp(-t / T2)


def sum_of_squared_residuals(params, times, signal):
    """Sum of squared residuals objective function"""
    A, B, T1star = params
    model = intensity_model_t1(times, A, B, T1star)
    return np.sum((signal - model) ** 2)


def solve_single_t1_faster_solver(signal, times):
    """Solves for T1 using the signal and times of a single voxel."""
    init_guess = [6, 3, 1000]
    bounds = ((0, None), (0, None), (0, None))

    res = optimize.minimize(
        sum_of_squared_residuals,
        init_guess,
        args=(times, signal),
        method="SLSQP",
        bounds=bounds,
        tol=1e-8,
    )

    A, B, T1star = res.x

    return T1star * (B / A - 1)


if __name__ == "__main__":
    from dataloading import MappingSingleCoilDataset, MappingDatapoint
    from matplotlib import pyplot as plt

    mc = True

    folder = "/home/mt3019/biomedia/vol/biodata/data/CMRxRecon 2023"
    dataset = MappingSingleCoilDataset(
        folder,
        True,
        4,
        debug=True,
        progress_bar=False,
        number_coils="multi" if mc else "single",
    )
    example: MappingDatapoint = dataset[0][1]

    if example.t1_map_roi is not None:
        t1_mask = example.t1_map_roi[..., 0]
        t1_mask = t1_mask.T.numpy()[::-1, ::-1]
        t1_min_x = np.min(np.where(t1_mask > 0)[0]) - 10
        t1_max_x = np.max(np.where(t1_mask > 0)[0]) + 10
        t1_min_y = np.min(np.where(t1_mask > 0)[1]) - 10
        t1_max_y = np.max(np.where(t1_mask > 0)[1]) + 10
    else:
        t1_min_x = 0
        t1_max_x = example.t1_kspace.shape[-2]
        t1_min_y = 0
        t1_max_y = example.t1_kspace.shape[-1]

    if example.t2_map_roi is not None:
        t2_mask = example.t2_map_roi[..., 0]
        t2_mask = t2_mask.T.numpy()[::-1, ::-1]
        t2_min_x = np.min(np.where(t2_mask > 0)[0]) - 10
        t2_max_x = np.max(np.where(t2_mask > 0)[0]) + 10
        t2_min_y = np.min(np.where(t2_mask > 0)[1]) - 10
        t2_max_y = np.max(np.where(t2_mask > 0)[1]) + 10
    else:
        t2_min_x = 0
        t2_max_x = example.t2_kspace.shape[-2]
        t2_min_y = 0
        t2_max_y = example.t2_kspace.shape[-1]

    start_time = time.time()
    t1_map = compute_t1_mapping(example.t1_times.numpy(), example.t1_kspace, multi_coil=mc)
    print(f"Time taken T1: {time.time() - start_time}")

    start_time = time.time()
    t2_map = compute_t2_mapping(
        example.t2_times.numpy(),
        example.t2_kspace.numpy(),
        iterative=False,
        multi_coil=mc,
    )
    print(f"Time taken T2: {time.time() - start_time}")

    new = t1_map[0].copy()
    new = new[t1_min_x:t1_max_x, t1_min_y:t1_max_y]
    plt.imshow(new, cmap="jet")
    plt.colorbar()
    plt.savefig("T1map.png")

    new = t2_map[0].copy()
    new = new[t2_min_x:t2_max_x, t2_min_y:t2_max_y]
    new[new > 2.5e2] = 0
    plt.imshow(new, cmap="jet")
    plt.colorbar()
    plt.savefig("T2map.png")
