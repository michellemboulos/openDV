import cv2
import edgeiq

def main():
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)


    image_paths = sorted(list(edgeiq.list_images("images/")))
    print("Images:\n{}\n".format(image_paths))

    with edgeiq.Streamer(
            queue_depth=len(image_paths), inter_msg_time=4) as streamer:
        for image_path in image_paths:
            # Load image from disk
            image = cv2.imread(image_path)

            results = obj_detect.detect_objects(image, confidence_level=.5)
            image = edgeiq.markup_image(
                    image, results.predictions, colors=[(255, 255, 255)])

            # Generate text to display on streamer
            text = ["<b>Model:</b> {}".format(obj_detect.model_id)] 
            text.append("<b>Inference time:</b> {:1.3f} s".format(results.duration))
            text.append("<b>Objects:</b>")

            for prediction in results.predictions:
                text.append("{}: {:2.2f}%".format(
                    prediction.label, prediction.confidence * 100))
            if image_path == 'images/example_08.jpg':
                text.append("<br><br><b><em>Hello, World!</em></b>")

            streamer.send_data(image, text)
        streamer.wait()

    print("Program Ending")


if __name__ == "__main__":
    main()
