import os


def get_m(message, subject=None, type_message='INFO'):
    try:
        if not isinstance(subject, str):
            subject = ' '.join(subject)
        return f'{type_message} - {subject}: {message}'
    except:
        subject = None
        return f'{type_message}: {message}'

