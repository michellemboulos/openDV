import cv2
import edgeiq
"""
Use two image classifiers to classify a batch of human face images by
age and gender. Different images can be used by updating the files in
the *images/* directory. Note that when developing for a remote device,
removing images in the local *images/* directory won't remove images
from the device. They can be removed using the `aai app shell` command
and deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""


def main():
    try:
        classifier1 = edgeiq.Classification("alwaysai/gendernet")
        classifier2 = edgeiq.Classification("alwaysai/agenet")

        classifier1.load(edgeiq.Engine.DNN)
        print("Engine 1: {}".format(classifier1.engine))
        print("Accelerator 1: {}\n".format(classifier1.accelerator))
        print("Model 1:\n{}\n".format(classifier1.model_id))
        print("Labels:\n{}\n".format(classifier1.labels))

        classifier2.load(edgeiq.Engine.DNN)
        print("Engine 2: {}".format(classifier2.engine))
        print("Accelerator 2: {}\n".format(classifier2.accelerator))
        print("Model 2:\n{}\n".format(classifier2.model_id))
        print("Labels:\n{}\n".format(classifier2.labels))

        image_paths = sorted(list(edgeiq.list_images("images/")))
        print("Images:\n{}\n".format(image_paths))

        with edgeiq.Streamer(
                queue_depth=len(image_paths), inter_msg_time=3) as streamer:
            for image_path in image_paths:
                image_display = cv2.imread(image_path)
                image = image_display.copy()

                results1 = classifier1.classify_image(
                        image, confidence_level=.95)
                results2 = classifier2.classify_image(image)

                # Generate text to display on streamer
                text = ["Model 1: {}".format(classifier1.model_id)]
                text.append("Model 2: {}".format(classifier2.model_id))
                text.append("Inference time: {:1.3f} s".format(
                    results1.duration + results2.duration))

                # Find the index of highest confidence
                if len(results1.predictions) > 0:
                    top_prediction1 = results1.predictions[0]
                    top_prediction2 = results2.predictions[0]
                    text1 = "Classification: {}, {:.2f}%".format(
                            top_prediction1.label,
                            top_prediction1.confidence * 100)
                    text2 = "Classification: {}, {:.2f}%".format(
                            top_prediction2.label,
                            top_prediction2.confidence * 100)
                else:
                    text1 = "Can not classify this image, confidence under " \
                            "95 percent for Gender Identification"
                    text2 = None
                # Show the image on which inference was performed with text
                cv2.putText(
                        image_display, text1, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                text.append(text1)
                if text2 is not None:
                    cv2.putText(
                            image_display, text2, (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    text.append(text2)

                streamer.send_data(image_display, text)
            streamer.wait()

    finally:
        print("Program Ending")


if __name__ == "__main__":
    main()
