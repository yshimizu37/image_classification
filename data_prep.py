import os
import argparse


def image_count(dataset_path):
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # 学習用の猫画像のディレクトリ
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # 学習用の犬画像のディレクトリ
    validation_cats_dir = os.path.join(
        validation_dir, 'cats')  # 検証用の猫画像のディレクトリ
    validation_dogs_dir = os.path.join(
        validation_dir, 'dogs')  # 検証用の犬画像のディレクトリ

    """### データの理解

  学習および検証ディレクトリの中にある猫と犬の画像の数を見てみましょう:
  """

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)


def main() -> None:
    parser = argparse.ArgumentParser(description='説明')

    parser.add_argument('-d', '--datadir', default="/mnt/dataset", help='dataset directory')

    args = parser.parse_args()

    image_count(dataset_path=args.datadir)

if __name__ == "__main__":
    main()