from treebanks.NodeOperations import children, isCo, realRel, realNode, findClosest
import unicodedata as ud

def removePaddedEllipsis(tree,wordid='wordid'):
    removeNode = None
    for node in reversed(tree):
        if 'artificial' in node and node['relation'] == 'UNLINKED':
            c = children(node,tree)
            if len(c) == 0:   
                removeNode = node
                break
    if removeNode is not None:
        tree.remove(removeNode)
        removePaddedEllipsis(tree,wordid)

def modifyPragueCoordination(tree,wordid='wordid',print_errors=True):
    for node in tree:
        if node['relation'] == 'COORD':
            co = []
            sh_mod = []
            coordination_rel = None
            for c in children(node,tree):
                if not 'new_rel' in c and c['relation'].endswith('_CO'):
                    coordination_rel = c['relation']
                    break
            for c in children(node,tree):
                if isCo(c,tree) and not (coordination_rel is not None and not realRel(c,tree,print_errors) == coordination_rel):
                    co.append(c)
                else:
                    sh_mod.append(c)
            if len(co) >= 2:
                for i in range(1,len(co)):
                    co[i]['head'] = realNode(co[i-1],tree,print_errors=print_errors)
            if len(co) == 0:
                id = node[wordid]
                if print_errors:
                    print(f'modifyPragueCoordination {id}: no co nodes found')
            else:
                firstCo = realNode(co[0],tree,print_errors=print_errors)
                firstCo['new_rel'] = firstCo['relation'].replace('_CO','')
                co[0]['head'] = node['head']
            coordinators = []
            coordinators.append(node)
            for mod in sh_mod:
                if ud.normalize('NFC',mod['lemma']) in ['καί','μήτε','οὔτε','ἤ','οὐδέ','ἀλλά','ἔπειτα','μηδέ','οὐδέ','εἴτε','εἶτα'] and mod['relation'] == 'AuxY':
                    coordinators.append(mod)
            for coordinator in coordinators:
                found = False
                for coordinate in co:
                    if tree.index(coordinate) > tree.index(coordinator):
                        coordinator['head'] = coordinate
                        coordinator['relation'] = 'AuxY'
                        found = True
                        break
                    if not found:
                        if len(co) != 0:
                            coordinator['head'] = co[len(co)-1]
                        coordinator['relation'] = 'AuxY'
            sh_mod = [mod for mod in sh_mod if mod not in coordinators]
            coordinators.clear()
            for mod in sh_mod:
                if ud.normalize('NFC',mod['lemma']) in ['δέ','τε'] and mod['relation'] == 'AuxY':
                    coordinators.append(mod)
            for coordinator in coordinators:
                closest = findClosest(coordinator,co,tree)
                if closest is not None:
                    coordinator['head'] = closest
                    coordinator['relation'] = 'AuxY'
                else:
                    id = node[wordid]
                    if print_errors:
                        print(f'modifyPragueCoordination {id}: no closest coordinator found')
            sh_mod = [mod for mod in sh_mod if mod not in coordinators]
            for mod in sh_mod:
                if len(coordinators)>0:
                    mod['head'] = coordinators[0]
    for node in tree:
        if 'new_rel' in node:
            node['relation'] = node['new_rel']
            del node['new_rel']
        elif node['relation'].endswith('_CO'):
            node['relation'] = 'CO'