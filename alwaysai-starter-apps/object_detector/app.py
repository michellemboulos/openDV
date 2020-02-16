import cv2
import edgeiq
"""
Use object detection to detect objects on a batch of images. The types
of objects detected can be changed by selecting different models.
Different images can be used by updating the files in the *images/*
directory. Note that when developing for a remote device, removing
images in the local *images/* directory won't remove images from the
device. They can be removed using the `aai app shell` command and
deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""


def main():
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    image_paths = sorted(list(edgeiq.list_images("images/")))
    print("Images:\n{}\n".format(image_paths))

    with edgeiq.Streamer(
            queue_depth=len(image_paths), inter_msg_time=3) as streamer:
        for image_path in image_paths:
            # Load image from disk
            image = cv2.imread(image_path)

            results = obj_detect.detect_objects(image, confidence_level=.5)
            image = edgeiq.markup_image(
                    image, results.predictions, colors=obj_detect.colors)

            # Generate text to display on streamer
            text = ["Model: {}".format(obj_detect.model_id)]
            text.append("Inference time: {:1.3f} s".format(results.duration))
            text.append("Objects:")

            for prediction in results.predictions:
                text.append("{}: {:2.2f}%".format(
                    prediction.label, prediction.confidence * 100))

            streamer.send_data(image, text)
        streamer.wait()

    print("Program Ending")


if __name__ == "__main__":
    main()
