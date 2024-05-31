import shutil
import cv2
import typing
import numpy as np
import os

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs

def startPrediction():
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join("..", "handwritten_to_digit",
                                 "input_sentences")
    path_input = os.path.abspath(os.path.join(current_directory,
                                              relative_path))
    output_path = os.path.join("..", "..", "static", "prediction_text")
    path_connect = os.path.abspath(os.path.join(current_directory, output_path))
    temp_path = os.path.join("saved_model", "configs.yaml")
    path_config = os.path.abspath(os.path.join(current_directory, temp_path))
    temp1_path = os.path.join("saved_model")
    path_model = os.path.abspath(os.path.join(current_directory, temp1_path))
    open(os.path.join(path_connect, "output.txt"), "w").close()
    configs = BaseModelConfigs.load(path_config)
    model = ImageToWordModel(configs.vocab, input_name="input",
                         model_path=path_model)


    df = pd.read_csv(os.path.join(path_input, "input_img_paths.csv")).values.tolist()
    f = open(os.path.join(path_connect, "output.txt"), "w")
    for image_path in tqdm(df):
        new_img = image_path[0]
        print(new_img)
        image = cv2.imread(new_img)
        prediction_text = model.predict(image)
        f.write(prediction_text + '\n')
        print("Prediction: ", prediction_text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    shutil.rmtree(os.path.join(path_input, 'input_lines_in_jpg'))
    os.mkdir(os.path.join(path_input, 'input_lines_in_jpg'))
    f.close()
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], input_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
        self.input_name = input_name
        self.input_shape = (96, 1408, 3)


    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image,
                                                             *self.input_shape[
                                                              :2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text



if __name__ == "__main__":
    startPrediction()

