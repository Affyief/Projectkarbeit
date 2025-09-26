from pypylon import pylon
import cv2

def main():
    # Create an instant camera object with the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Start grabbing images
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000)  # Timeout in ms

        if grab_result.GrabSucceeded():
            # Access the image data as numpy array
            img = grab_result.Array

            # Display image using OpenCV
            cv2.imshow("Basler Camera", img)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grab_result.Release()

    camera.StopGrabbing()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
