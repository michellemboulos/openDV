import logging
import time
import delivery
import edgeiq


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN_OPENVINO, accelerator=edgeiq.Accelerator.MYRIAD)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = [""]
                for ind, pose in enumerate(results.poses):
                    if all(y < pose.key_points['Nose'] for y in (pose.key_points['Right Elbow'], pose.key_points['Left Elbow'])):
                        delivery.send("DS identified. Message sent.")
                        print("Message Delivered.")
                        break
                    break
                        #print("TWILIO CODE SUPPOSED TO GO HERE")

                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
