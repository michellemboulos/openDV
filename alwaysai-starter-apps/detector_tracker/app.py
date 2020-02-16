import time
import edgeiq
"""
Use object detection and tracking to follow objects as they move across
the frame. Detectors are resource expensive, so this combination
reduces stress on the system, increasing the resulting bounding box output
rate. The detector is set to execute every 30 frames, but this can be
adjusted by changing the value of the `detect_period` variable.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""


def main():
    # The current frame index
    frame_idx = 0
    # The number of frames to skip before running detector
    detect_period = 30

    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    tracker = edgeiq.CorrelationTracker(max_objects=5)
    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                frame = video_stream.read()
                predictions = []
                if frame_idx % detect_period == 0:
                    results = obj_detect.detect_objects(
                            frame, confidence_level=.5)
                    # Generate text to display on streamer
                    text = ["Model: {}".format(obj_detect.model_id)]
                    text.append(
                            "Inference time: {:1.3f} s".format(
                                results.duration))
                    text.append("Objects:")

                    # Stop tracking old objects
                    if tracker.count:
                        tracker.stop_all()

                    predictions = results.predictions
                    for prediction in predictions:
                        text.append("{}: {:2.2f}%".format(
                            prediction.label, prediction.confidence * 100))
                        tracker.start(frame, prediction)
                else:
                    if tracker.count:
                        predictions = tracker.update(frame)

                frame = edgeiq.markup_image(
                        frame, predictions, show_labels=True,
                        show_confidences=False, colors=obj_detect.colors)
                streamer.send_data(frame, text)
                frame_idx += 1
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        tracker.stop_all()
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
