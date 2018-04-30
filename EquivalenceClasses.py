"""
Algorithm to categorize elements into equivalence classes
http://code.activestate.com/recipes/499354-equivalence-partition/
"""
def equivalence_partition( iterable, relation, verbose=False ):
    """
    Partitions a set of objects into equivalence classes\n",
    Args:\n",
        iterable: collection of objects to be partitioned\n",
        relation: equivalence relation. I.e. relation(o1,o2) evaluates to True\n",
            if and only if o1 and o2 are equivalent\n",
    Returns: classes, partitions\n",
        partitions: A dictionary mapping objects to equivalence classes\n",
    """
    class_representatives = []
    partitions = {}
    try:
        for counter, obj in enumerate(iterable): # for each object
        # find the class it is in\n",
            found = False
            for k, element in enumerate(class_representatives):
                if relation( element, obj ): # is it equivalent to this class?
                    partitions[counter] = k
                    found = True
                    break
            if not found: # it is in a new class\n",
                class_representatives.append( obj )
                partitions[counter] = len(class_representatives) - 1
            if verbose:
                print('Tested {:d} elements and found {:d} classes'.format(counter+1,len(class_representatives)))
    except KeyboardInterrupt:
        pass
    return class_representatives, partitions
