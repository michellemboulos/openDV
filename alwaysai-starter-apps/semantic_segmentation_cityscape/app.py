import cv2
import edgeiq
"""
Use semantic segmentation to determine a class for each pixel of an image.
The classes of objects detected can be changed by selecting different models.
This particular starter application uses a model on the cityscape
dataset (https://www.cityscapes-dataset.com/).
The Cityscapes Dataset focuses on semantic understanding of urban street scenes,
and is a favorite dataset for building autonomous car machine learning models.

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
    semantic_segmentation = edgeiq.SemanticSegmentation(
            "alwaysai/enet")
    semantic_segmentation.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(semantic_segmentation.engine))
    print("Accelerator: {}\n".format(semantic_segmentation.accelerator))
    print("Model:\n{}\n".format(semantic_segmentation.model_id))
    print("Labels:\n{}\n".format(semantic_segmentation.labels))

    image_paths = sorted(list(edgeiq.list_images("images/")))
    print("Images:\n{}\n".format(image_paths))

    with edgeiq.Streamer(
            queue_depth=len(image_paths), inter_msg_time=3) as streamer:
        for image_path in image_paths:
            image = cv2.imread(image_path)

            results = semantic_segmentation.segment_image(image)

            # Generate text to display on streamer
            text = ["Model: {}".format(semantic_segmentation.model_id)]
            text.append("Inference time: {:1.3f} s".format(results.duration))
            text.append("Legend:")
            text.append(semantic_segmentation.build_legend())

            mask = semantic_segmentation.build_image_mask(results.class_map)
            blended = edgeiq.blend_images(image, mask, alpha=0.5)

            streamer.send_data(blended, text)
            streamer.wait()

        print("Program Ending")


if __name__ == "__main__":
    main()
