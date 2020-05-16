_global_dict = None


def init():
    """
    init a global dict to save global vars
    :return:
    """
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    """ set a global var """
    _global_dict[key] = value


def get_value(key):
    """ get a global var """
    try:
        return _global_dict[key]
    except KeyError:
        print("ERROR: This global var doesn't exist!")
        exit(1)