# prepare dependencies by installing in
# your environment with the following command:
# pip install cmake opencv-python imutils dlib

# import the necessary packages
import scipy.spatial.distance as dist  # para a euclidean distance entre landmarks  # noqa: E501
from imutils import face_utils
import argparse
import imutils  # image processing functions
import dlib  # detect and localize landmks
import cv2
import os  # para encontrar todos os ficheiros numa diretoria
import time


# define constant to indicate blink (threshold)
EAR_THRESH = 0.25
YAWN_THRESH = 17

# define constant for the number of consecutive frames the
# eye/lip-dist must be low/high to raise alert
EAR_CONSEC_FRAMES = 48
YAWN_CONSEC_FRAMES = 100


def eye_aspect_ratio(eye):
    """returns the EAR (eye aspect ratio)"""

    # compute the euclidean distances between the two sets
    # of vertical eye landmarks (x, y)-coordinates
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    h = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (v1 + v2) / (2.0 * h)

    # return the eye aspect ratio
    return ear


def mouth_open(topLip, bottomLip):
    """returns the lip distance"""

    # find the lip center - maybe improvement: lip average
    topLipCenter = topLip[2]
    bottomLipCenter = bottomLip[2]

    # compute the euclidean distance between the top and
    # bottom lip centers (x, y)-coordinates
    lip_distance = dist.euclidean(topLipCenter, bottomLipCenter)

    return lip_distance


def cli():
    """construct the argument parse and parse the arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--shape_predictor",
        default="./shape_predictor_68_face_landmarks.dat",
        help="path to facial landmark predictor"
    )
    ap.add_argument(
        "-d", "--source_dir",
        default=".",
        help="Directory containing videos to analyze."
    )
    ap.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory where to store collected features."
    )
    args = ap.parse_args()

    main(args.shape_predictor, args.source_dir, args.output_dir)


def annotate_frame(
    frame, leftEyeHull, rightEyeHull,
    innerMouth, ear, check_ear,
    lip, check_lip
):
    # visualize both eyes on the frame
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    # visualize mouth on the frame
    cv2.drawContours(frame, [innerMouth], -1, (0, 255, 0), 1)

    # draw the computed EAR on the frame
    cv2.putText(
        frame,
        "EAR: {:.2f}".format(ear),
        (245, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2
    )

    # draw the computed lip distance on the frame
    cv2.putText(
        frame,
        "Lip Distance: {:.2f}".format(lip),
        (245, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2
    )

    # draw warnings if checks are flagged
    if check_ear:
        cv2.putText(
            frame,
            "LOW EAR ALERT!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

    if check_lip:
        cv2.putText(
            frame,
            "YAWN ALERT!",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )


def main(predictor_path, source_path, output_path):
    # initialize dlib's face detector (HOG-based)
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()

    # create the facial landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    # LANDMARKS INDEXES for different face parts
    # grab the indexes of the facial landmarks for the left and right eyes
    (lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # grab the indexes of the facial landmarks for the inner mouth
    (iMouthStart, iMouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    # get the videos from a folder
    _, _, files = next(os.walk(source_path))
    videos = list(filter(lambda x: '.mp4' in x, files))
    print("[INFO] accessing Database...")

    # loop over videos in a shorter folder (for testing)
    for file in videos:
        index = 0
        output_file = open(
            os.path.join(
                output_path,
                os.path.splitext(file)[0]+".csv"
            ),
            'w'
        )
        #output_file.write("frame,left_eye,right_eye,lip_distance\n")
        output_file.write("frame,leftEye,rightEye,lip_distance\n")

        # initialize the frame counter
        bCOUNTER = 0  # blink-EAR
        yCOUNTER = 0  # yawn

        video = cv2.VideoCapture(os.path.join(source_path, file))
        time.sleep(1.0)
        # loop over frames from the video
        while True:
            frame = video.read()
            if not frame[0]:  # frame 0 is true ou false
                print("no frame - break")
                break

            # resize frame
            frame = imutils.resize(frame[1], width=450)  # [1] bc tuple

            # gray (eliminate the '3')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            # possibilidade de detecao tb da cara do copiloto
            rects = detector(gray_frame, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region
                shape = predictor(gray_frame, rect)

                # convert the facial landmark (x, y)-coordinates
                # to a numpy array
                shape = face_utils.shape_to_np(shape)

                # EYES-----

                # extract the left and right eye coordinates
                leftEye = shape[lEyeStart:lEyeEnd]
                rightEye = shape[rEyeStart:rEyeEnd]

                # compute the convex hull for the left and right eye
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                # MOUTH-----

                # extract the inner lips coordinates
                topLip = shape[60:65]
                bottomLip = shape[64:68]

                # extract the inner mouth coordinates
                innerMouth = shape[iMouthStart:iMouthEnd]

                # compute the convex hull for the inner mouth
                # innerMouthHull = cv2.convexHull(innerMouth)

                # -----BLINK-EAR-----

                # compute the eye aspect ratio for both eyes
                # using both eye coordinates
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio for both eyes
                eye = (leftEAR + rightEAR) / 2.0

                # -----YAWN-----

                # compute the lip distance
                lip_distance = mouth_open(topLip, bottomLip)

                # Write to file
                #output_file.write(
                #    "{index},{leftEAR},{rightEAR},{lip_distance}\n"
                #)

                output_file.write(f"{index},{leftEAR},{rightEAR},{lip_distance}\n")

                # ----------checks

                if eye < EAR_THRESH:
                    bCOUNTER += 1
                else:
                    # reset the frame counter
                    bCOUNTER = 0

                if lip_distance > YAWN_THRESH:
                    yCOUNTER += 1
                else:
                    # reset the frame counter
                    yCOUNTER = 0

                annotate_frame(
                    frame,
                    leftEyeHull, rightEyeHull,
                    innerMouth,
                    eye,
                    bCOUNTER >= EAR_CONSEC_FRAMES,
                    lip_distance,
                    yCOUNTER >= YAWN_CONSEC_FRAMES
                )

            index += 1
            # visualize the frame
            cv2.imshow("Frame", frame)

            # keyboard action!
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, break from the while loop
            if key == ord("q"):
                break


        # end while (1i)

        output_file.close()
    # end for (0i)

    # do a bit of cleaning
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
