from RPi.GPIO import HIGH, LOW, BCM, output, setup
from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput, logUsage

from argparse import ArgumentParser, RawTextHelpFormatter
from sys import argv, exit

MOVE_BACK = 18
MOVE_FORWARD = 17
MOVE_LEFT = 16
MOVE_RIGHT = 20
MOVE_UP = 21
MOVEMENTS = [MOVE_BACK, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]

def back:
	output(MOVE_BACK, LOW)
	print("back")

def forward:
	output(MOVE_FORWARD, LOW)
	print("forward")

def left:
	output(MOVE_LEFT, LOW)
	print("left")

def right:
	output(MOVE_RIGHT, LOW)
	print("right")

def up:
	output(MOVE_UP, LOW)
	print("up")

def reset:
	for m in MOVEMENTS:
		output(m, HIGH)
	print("nothing")

def init(opt):
	# load the object detection network
	net = detectNet(opt.network, argv, opt.threshold)
	is_headless = ["--headless"] if argv[0].find('console.py') != -1 else [""]
	# create video sources & outputs
	video_source = videoSource(opt.input_URI, argv = argv)
	video_output = videoOutput(opt.output_URI, argv = argv + is_headless)
	
	#setup GPIO pins
	setmode(BCM) #RaspPi pin numbering
	
	for m in MOVEMENTS:
		setup(m, OUT, initial = HIGH)
	return net, video_source, video_output

def loop(opt, net, video_source, video_output):
	# process frames until the user exits
	# capture the next image
	img = video_source.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay = opt.overlay)

	# print the detections
	# print(f"detected {len(detections):d} objects in image")

	# check for detections, otherwise nothing

	if(len(detections) > 0):
		detection, *_ = detections
		print("object detected")
		index = detection.ClassID
		confidence = (detection.Confidence)

		# print index of item, width and horizonal location

		print(index, detection.Width, detection.Center[0], confidence)

		# look for detections

		if (index == 1 and confidence > 0.9): back()
		elif (index == 2 and confidence > 0.7): forward()
		elif (index == 3 and confidence > 0.7): left()
		elif (index == 4 and confidence > 0.7): right()
		elif (index == 5 and confidence > 0.7): up()
	else:
		reset()	# nothing is detected
	
	# render the image
	video_output.Render(img)
	
	# update the title bar
	video_output.SetStatus(
		f"{opt.network:s} | Network {net.GetNetworkFPS():.0f} FPS")
	
	# print out performance info
	#net.PrintProfilerTimes()
	
	# exit on input/output EOS
	if not video_source.IsStreaming() or not video_output.IsStreaming():
		break

if __name__ == "__main__":
	# parse the command line
	parser = ArgumentParser(
		description = "Locate objects in a live camera stream using an object detection DNN.",
		formatter_class = RawTextHelpFormatter,
		epilog = detectNet.Usage()
			+ videoSource.Usage()
			+ videoOutput.Usage()
			+ logUsage()
	)
	options = [
		("input_URI", {"type": str, "default": '', "nargs": '?', "help": "URI of the input stream"}),
		("output_URI", {"type": str, "default": '', "nargs": '?', "help": "URI of the output stream"}),
		("--network", {"type": str, "default": "ssd-mobilenet-v2", "help": "pre-trained model to load (see below for options)"}),
		("--overlay", {"type": str, "default": "box,labels,conf", "help": "detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'"}),
		("--threshold", {"type": float, "default": 0.5, "help": "minimum detection threshold to use"})]
	for var, option in options:
		parser.add_argument(var, **option)
	try:
		opt = parser.parse_known_args()[0]
	except:
		print('')
		parser.print_help()
		exit(0)
	net, video_source, video_output = init(opt)
	while True:
		loop(opt, net, video_source, video_output)
