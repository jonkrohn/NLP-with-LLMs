## This code was created by Shaan Khosla. See original here: https://github.com/shaankhosla/NLP_with_LLMs/blob/main/generate_data.py
## He adapted large portions from https://docs.mosaicml.com/projects/streaming/en/stable/examples/synthetic_nlp.html

from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import json, os


ones = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    + "fifteen sixteen seventeen eighteen nineteen"
).split()

tens = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def say(i: int) -> List[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ["negative"] + say(-i)
    elif i <= 19:
        return [ones[i]]
    elif i < 100:
        return [tens[i // 10 - 2]] + ([ones[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [ones[i // 100], "hundred"] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return say(i // 1_000) + ["thousand"] + (say(i % 1_000) if i % 1_000 else [])
    elif i < 1_000_000_000:
        return (
            say(i // 1_000_000)
            + ["million"]
            + (say(i % 1_000_000) if i % 1_000_000 else [])
        )
    else:
        assert False


def get_random_number() -> int:
    """Pick a random number the way humans would.

    Picked numbers are positively skewed, exponentially distributed (good for curriculum learning).

    Returns:
        int: The number.
    """
    sign = (np.random.random() < 0.8) * 2 - 1
    mag = 10 ** np.random.uniform(1, 4) - 10
    return sign * int(mag**2)


def get_numbers(num_train: int, num_val: int) -> Tuple[List[int], List[int]]:
    """Get two non-overlapping splits of unique random numbers.

    Because the distribution is exponential, we are unlikely to run out of numbers.

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.

    Returns:
        Tuple[List[int], List[int]]: The two generated splits.
    """
    total = num_train + num_val
    numbers = set()
    bar = tqdm(total=total, leave=False)
    while len(numbers) < total:
        was = len(numbers)
        numbers.add(get_random_number())
        bar.update(len(numbers) - was)
    numbers = list(numbers)
    np.random.shuffle(numbers)
    return numbers[:num_train], numbers[num_train:]


def generate_samples(numbers: List[int]) -> List[Dict[str, Any]]:
    """Generate samples from a list of numbers.

    Args:
        numbers (List[int]): The numbers.

    Returns:
        List[Dict[str, Any]]: The corresponding samples.
    """
    samples = []
    for num in numbers:
        words = " ".join(say(num))
        sample = {"number": num, "words": words}
        samples.append(sample)
    return samples


def get_dataset(
    num_train: int, num_val: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate a number-saying dataset of the given size.

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: The two generated splits.
    """
    train_nums, val_nums = get_numbers(num_train, num_val)
    train_samples = generate_samples(train_nums)
    val_samples = generate_samples(val_nums)
    return train_samples, val_samples


def create_folder_structure():
    if not os.path.isdir("./data/"):
        os.mkdir("./data/")
    if not os.path.isdir("./data/train/"):
        os.mkdir("./data/train/")
    if not os.path.isdir("./data/val/"):
        os.mkdir("./data/val/")

    for f in os.listdir("./data/train/"):
        os.remove(os.path.join("./data/train", f))
    for f in os.listdir("./data/val/"):
        os.remove(os.path.join("./data/val", f))


def main(num_train: int, num_val: int):
    print(f"Generating synthetic dataset ({num_train} train, {num_val} val)...")
    train_samples, val_samples = get_dataset(num_train, num_val)

    create_folder_structure()

    for i in range(len(train_samples)):
        with open(f"./data/train/{i}.json", "w") as outfile:
            json.dump(train_samples[i], outfile)

    for j in range(len(val_samples)):
        with open(f"./data/val/{j}.json", "w") as outfile:
            json.dump(val_samples[j], outfile)


if __name__ == "__main__":
    # Number of training and validation samples
    num_train_samples = 10_000  # 10k samples
    num_val_samples = 2000  # 2k samples

    # Create the samples.
    main(num_train_samples, num_val_samples)
