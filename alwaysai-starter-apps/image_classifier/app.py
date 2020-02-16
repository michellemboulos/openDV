import cv2
import edgeiq
"""
Use image classification to classify a batch of images. The
classification labels can be changed by selecting different models.
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
    classifier = edgeiq.Classification("alwaysai/googlenet")
    classifier.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(classifier.engine))
    print("Accelerator: {}\n".format(classifier.accelerator))
    print("Model:\n{}\n".format(classifier.model_id))
    print("Labels:\n{}\n".format(classifier.labels))

    image_paths = sorted(list(edgeiq.list_images("images/")))
    print("Images:\n{}\n".format(image_paths))

    with edgeiq.Streamer(
            queue_depth=len(image_paths), inter_msg_time=3) as streamer:
        for image_path in image_paths:
            image_display = cv2.imread(image_path)
            image = image_display.copy()

            results = classifier.classify_image(image)

            # Generate text to display on streamer
            text = ["Model: {}".format(classifier.model_id)]
            text.append("Inference time: {:1.3f} s".format(results.duration))

            if results.predictions:
                image_text = "Label: {}, {:.2f}".format(
                        results.predictions[0].label,
                        results.predictions[0].confidence)
                cv2.putText(
                        image_display, image_text, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for idx, prediction in enumerate(results.predictions[:5]):
                    text.append("{}. label: {}, confidence: {:.5}".format(
                        idx + 1, prediction.label, prediction.confidence))
            else:
                text.append("No classification for this image.")

            streamer.send_data(image_display, text)
        streamer.wait()

    print("Program Ending")


if __name__ == "__main__":
    main()
