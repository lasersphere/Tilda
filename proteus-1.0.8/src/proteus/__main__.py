import pathlib
import argparse
import proteus


parser = argparse.ArgumentParser()  # TODO description=
parser.add_argument(
	'config_file',
	action='store',
	nargs='?',
	default='proteus.json',
	type=pathlib.Path,
	help='The json file with the PROTEUS configuration to use'
)
config = proteus.read_configuration(
	str(parser.parse_args().config_file),
	relevant_methods=(proteus.Instance, proteus.Window)
)
with proteus.Instance(**config['Instance']) as instance:
	if 'Window' in config:
		window = proteus.Window(instance, **config['Window'])
		# the variable window is very necessary: otherwise the window might get garbage collected
		window.wait_for_window()
	# TODO some alternate windowless interaction/input mode?
