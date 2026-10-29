"""Independent final byte observation, including zero-valued tail bytes."""


def require_output_bytes(actual,expected):
    if actual!=expected:
        for address in sorted(set(actual)|set(expected)):
            if actual.get(address)!=expected.get(address):
                raise AssertionError(f'output byte {address:#x}: observed={actual.get(address)} expected={expected.get(address)}')
        raise AssertionError('output byte map differs')
    return len(actual)
