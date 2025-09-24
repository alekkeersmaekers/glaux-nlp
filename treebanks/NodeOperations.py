import unicodedata as ud

def nonCoordinateHead(node,tree):
    if node['head'] is not None:
        if node['head']['relation'] == 'COORD':
            if node['relation'].endswith('_CO') or node['relation'] == 'COORD' or node['relation'] == 'AuxY':
                return nonCoordinateHead(node['head'],tree)
            else:
                for c in children(node['head'],tree):
                    if c['relation'].endswith('_CO'):
                        return c
                return nonCoordinateHead(node['head'],tree)
        else:
            return node['head']
    else:
        return None

def highestCoordinateUD(node,conjunct_relation='CO'):
    if node['head'] is None:
        return None
    elif node['head']['relation'] == conjunct_relation:
        return highestCoordinateUD(node['head'])
    else:
        return node['head']

def coordinationHeadUD(node,conjunct_relation='CO'):
    if node['head'] is None:
        return None
    elif node['head']['relation'] == conjunct_relation:
        return coordinationHeadUD(node['head'])
    else:
        return node['head']['head']

def children(node,tree):
    c = []
    for node2 in tree:
        if node2['head'] is not None and node2['head'] == node:
            c.append(node2)
    return c

def isCo(node,tree):
    if node['relation'] == 'AuxC' or node['relation'] == 'AuxP' or node['relation'] == 'COORD':
        for c in children(node,tree):
            if isCo(c,tree):
                return True
    elif node['relation'].endswith('_CO'):
        return True
    return False

def realRel(node,tree,print_errors=True):
    if node['relation'] == 'AuxC' or node['relation'] == 'AuxP':
        return realNode(node,tree,print_errors)['relation']
    elif node['relation'] == 'COORD':
        n_c = children(node,tree)
        if len(n_c) == 0:
            id = node['wordid']
            if print_errors:
                print(f'realRel: COORD with no children {id}')
            return None
        for c in n_c:
            rel = realRel(c,tree,print_errors)
            if rel is None:
                id = node['wordid']
                if print_errors:
                    print(f'realRel failed for node {id}')
                return None
            if rel.endswith('_CO'):
                return rel
        return realRel(n_c[0],tree,print_errors)
    else:
        return node['relation']
    
def realNode(node,tree,print_errors=True):
    if node['relation'] == 'AuxC' or node['relation'] == 'AuxP':
        nodes = []
        for c in children(node,tree):
            if not c['relation'] in ['AuxY','AuxZ','AuxG','AuxX','AuxK','PUNCT','MWE','MWE2']:
                nodes.append(realNode(c,tree,print_errors))
        if len(nodes) == 1:
            return nodes[0]
        elif len(nodes) == 0:
            id = node['wordid']
            if print_errors:
                print(f'realNode: no children found for node {id}')
            return node
        else:
            nodes2 = []
            for c in children(node,tree):
                if not ud.normalize('NFC',c['lemma']) == 'ὁ' and not c['relation'] in ['AuxY','AuxZ','AuxG','AuxX','AuxK','PUNCT','MWE','MWE2']:
                    nodes2.append(realNode(c,tree,print_errors))
            if len(nodes2) > 1:
                id = node['wordid']
                if print_errors:
                    print(f'realNode: multiple children found {id}, choosing first node {id}')
            return nodes2[0]
    else:
        return node
    
def findClosest(node,candidates,tree):
    min_dist = 1000
    closest = None
    for candidate in candidates:
        dist = abs(tree.index(node) - tree.index(candidate))
        if dist < min_dist:
            min_dist = dist
            closest = candidate
    return closest