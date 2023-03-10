from pickle import TRUE
import sys
import argparse
import os
from timeit import default_timer as timer
from yolo import YOLO, detect_video
from PIL import Image

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

test_dir = "C:/Users/15714/Desktop/Assignments/Special_Problem/Data" 
save_dir ="C:/Users/15714/Desktop/Assignments/Special_Problem/saved_result" 
testfiles= os.listdir(test_dir)

def detect_img(yolo):
    start = timer()
    for i in range(len(testfiles)):
        img = os.path.join(test_dir,testfiles[i])
        print ('file: ', img)
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save(save_dir + testfiles[i])
            r_image.show()
    end = timer()
    time_taken = end - start
    print("time taken : ", time_taken)
    yolo.close_session()



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=TRUE,default='C:/Users/15714/Desktop/Assignments/Special_Problem/vehicle_distance/Vid.mp4',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="C:/Users/15714/Desktop/Assignments/Special_Problem/vehicle_distance/savedresult.mp4",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
