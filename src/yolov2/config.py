from configparser import ConfigParser

parser = ConfigParser()
parser.read('config.ini')

# print(parser.get('data', 'labels'))
