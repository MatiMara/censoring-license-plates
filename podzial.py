import os
from PIL import Image
import shutil


def split_photo(img_name, vehicles_df=None):
    # print(vehicles_df)
    # Wczytanie wszystkich plików z folderu
    # input_img_path = f"C:/Users/mateu/Desktop/t1/{img_name}"
    input_img_path = img_name
    output_folder_path = "C:/Users/mateu/Desktop/t2"
    overlap_ratio = 0.15
    max_width = 1200
    max_height = 900

    # Usuń wszystkie pliki z folderu wyjściowego przed wykonaniem skryptu
    for filename in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Skopiuj wszystkie pliki z folderu wejściowego do folderu wyjściowego
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        output_image_path = os.path.join(output_folder_path, img_name.split('/')[-1])
        shutil.copyfile(input_img_path, output_image_path)

    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        # Wczytanie obrazu
        img = Image.open(input_img_path)

        # Sprawdzenie rozmiaru obrazu
        width, height = img.size
        if width > height:
            width_parts_n = width // (max_width/1.3) + 1
            height_parts_n = height // (max_height/1.3) + 1
        else:
            width_parts_n = width // (max_width/1.3) + 1
            height_parts_n = height // (max_height/1.3) + 1
        part_width = int(width // width_parts_n)
        part_height = int(height // height_parts_n)

        # Calculate the overlap distance in pixels
        overlap_width = int(part_width * overlap_ratio)
        overlap_height = int(part_height * overlap_ratio)

        # Obliczanie minimalnej szerokości i wysokości dla wyjściowych obrazów
        min_width = part_width + 2 * overlap_width
        min_height = part_height + 2 * overlap_height

        for row in range(int(height/part_height)):
            for col in range(int(width/part_width)):
                # Obliczenie pozycji wycinanego fragmentu z overlapem
                x1 = col * part_width - overlap_width if col > 0 else col * part_width
                y1 = row * part_height - overlap_height if row > 0 else row * part_height
                x2 = (col + 1) * part_width + overlap_width if col < int(width/part_width)-1 else (col + 1) * part_width
                y2 = (row + 1) * part_height + overlap_height if row < int(height/part_height)-1 else (row + 1) * part_height

                if vehicles_df is not None:
                    car = False
                    # Krzyżowanie koordynat - sprawdzenie, czy na wycinku jest pojazd
                    for index, vehicle in vehicles_df.iterrows():
                        if (not (x1 < vehicle["xmin"] and x2 < vehicle["xmin"])) and (not (y1 < vehicle["ymin"] and y2 < vehicle["ymin"])) and (not (x1 > vehicle["xmax"] and x2 > vehicle["xmax"])) and not (y1 > vehicle["ymax"] and y2 > vehicle["ymax"]):
                            car = True
                    if car is not True:
                        continue

                # Wycinanie i zapisywanie fragmentu
                part = img.crop((x1, y1, x2, y2))

                # Zmiana rozmiaru fragmentu do minimalnej szerokości i wysokości
                part = part.resize((min_width, min_height))

                output_filename = f"{os.path.splitext(img_name.split('/')[-1])[0]}_{row}_{col}_{x1}_{y1}_.jpg"
                output_image_path = os.path.join(output_folder_path, output_filename)
                part.save(output_image_path)


if __name__ == "__main__":
    imgname = "9.jpg"
    split_photo(imgname)