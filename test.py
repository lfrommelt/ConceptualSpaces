import xml.etree.ElementTree as ET
import os
import sys
sys.path.append("conceptual_spaces")
import cs.cs as space
from cs.weights import Weights
from cs.cuboid import Cuboid
from cs.core import Core
from cs.core import from_cuboids
from cs.concept import Concept
import matplotlib.pyplot as plt
import visualization.concept_inspector as ci



path = "Dataset\\prototypes.xml"
path2 = "Dataset\\exemplars.xml"


def print_database(node,indent=''):
    print(indent,node.tag,end=' ')
    try:
        print(node.attrib['name'],end=' ')
    except KeyError:
        pass
    if len(node):
        print()
        for subelem in node:
            print_database(subelem,indent+"  ")
    else:
        print(node.text)
    
def get_domains(xml_elem):
    domains = xml_elem.find('genericPhysicalDescription')
    if(not(domains)):
        return
    domain_dict={}
    try:
        supercategory = xml_elem.find('family').text
        if supercategory in ['feline','rodent','primate','cetacean']:
            supercategory='mammal'
        elif supercategory in ['crustacean']:
            supercategory = 'arthropod'
        elif supercategory in ['fruit', 'transport', 'furniture', 'book', 'building', 'musical_instrument', 'present', 'architectural_element']:
            supercategory = 'object'
        domain_dict['supercategory']=supercategory
    except AttributeError:
        domain_dict['supercategory']="object"
        
    for domain in domains:
        numerical_domain={}
        for subelem in domain:
            try:
                numerical_domain[subelem.tag]=float(subelem.text)
            except ValueError:
                pass
        if numerical_domain:
            domain_name=domain.tag
            
            if not domain_name=='size':
                pass
            elif(domain_name=='hasPart' or domain_name=='partOf'):
                pass
                '''domain_name = domain.get('name')
                domain_dict['n_'+domain_name]={'number_'+domain_name:value for key,value in zip(numerical_domain.keys(),numerical_domain.values())}'''
            else:
                domain_dict[domain_name]={key:value for key,value in zip(numerical_domain.keys(),numerical_domain.values())}
    return domain_dict

def xml_to_dict(path, exemplars = {}):
    tree = ET.parse(os.path.normpath(path))
    root = tree.getroot()
    for exemplar in root:
        #print(animal.attrib['name'])
        new_concept = get_domains(exemplar)
        if(new_concept):
            exemplars[exemplar.attrib['name']] = new_concept
    return exemplars

def is_consistent(concept):
    for domain in concept:
        if not domain=='supercategory':
            if not(len(concept[domain])==len(space._domains[domain])):
                return False
    return True
                
                
def concepts_into_space(concepts, domains={}, except_for=None, colors=['b','g','r','c','m','brown','rosybrown','darkslategrey','pink','grey']):
    if except_for:
        for concept in list(concepts.keys()):
            if concepts[concept]['supercategory']==except_for:
                del(concepts[concept])
                
    ordered_dict={}
    for example in concepts:
        if concepts[example]['supercategory'] in ordered_dict:
            ordered_dict[concepts[example]['supercategory']].append(example)
        else:
            ordered_dict[concepts[example]['supercategory']]=[example]
    '''for key in ordered_dict:
        print(key)'''
    #ordered_dict={'mammal':ordered_dict['mammal']}
    #print(ordered_dict)
    
    colors=iter(colors)
    concept_colors={}
    if(domains=={}):
        for concept in concepts.values():
            domains.update(concept)
        del(domains['supercategory'])
    #domains = mapping: domain->dimension indices
    domain_mapping = {}
    i=0
    for key in domains:
        domain_mapping[key]=list(range(i,i+len(domains[key])))
        i+=len(domains[key])

    dimension_names = []
    for domain in domains.values():
        dimension_names += [dim for dim in domain.keys()]
    space.init(len(dimension_names),domain_mapping,dimension_names)
    
    for category in ordered_dict:
        print('putting',category,'into space')
        cuboids=[]
        domains = {}
        for example in ordered_dict[category]:
            if not is_consistent(concepts[example]):
                print(example,' is inconsistent')
            else:
                subdomains={domain:space._domains[domain] for domain in concepts[example] if not domain == 'supercategory'}
                point=[concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values() for domain in space._domains]
                point=[p for dom in point for p in dom]
                #print(example)
                #print('point: ',point)
                cuboids.append(Cuboid([dim if not dim==float('inf') else -dim for dim in point], point, subdomains))
                domains.update(subdomains)
                #print(cuboids[-1]._p_max)
        #for cuboid in cuboids:
            #print('cuboid:',cuboid)
        core=from_cuboids(cuboids,domains)
        weights=Weights(space._def_dom_weights,space._def_dim_weights)
        concept=Concept(core, 1.0, 0.5, weights)
        space.add_concept(category,concept,next(colors))
        
    animal_concept=[]
    for concept in space._concepts.values():
        animal_concept+=concept._core._cuboids
    #print('animal cubs:\n',animal_concept)
    core=from_cuboids(animal_concept,space._domains)
    weights=Weights(space._def_dom_weights,space._def_dim_weights)
    concept=Concept(core, 1.0, 0.5, weights)
    space.add_concept('animal',concept,next(colors))
        
        
"""    for example in concepts:
        if concepts[example]['supercategory'] in ['mammal','reptile','bird','amphibian','insect']:
            consistent = True
            domains = {}
            for key in concepts[example]:
                if(not(key=='supercategory')):
                    domains[key]=space._domains[key]

            '''dimension_names = []

            for concept_domain in concepts[example].values():
                dimension_names += [dim for dim in domain.keys()]'''

            dimension_values=[]
            for domain in space._domains.keys():
                dimension_values.append([list(concepts[example].get(domain,{str(key):float("-inf") for key in range(len(space._domains[domain]))}).values())
                                     ,list(concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values())])
                if(not(len(dimension_values[-1][0])==len(space._domains[domain]))):
                    #print(domain+' in '+example+' is inconsistent')
                    consistent=False
                    break
                if(not(consistent)):
                    print('bah')


            if(consistent):
                p_min=[value for domain in dimension_values for value in domain[0]]
                p_max=[value for domain in dimension_values for value in domain[1]]

                #try:
                c_example = Cuboid(p_min, p_max, domains)
                s_example = Core([c_example], domains)
                w_example = Weights(space._def_dom_weights,space._def_dim_weights)
                concept = Concept(s_example, 1.0, 0.5, w_example)

                supercategory = concepts[example]['supercategory']
                if(supercategory in space._concepts):
                    space.add_concept(supercategory, space._concepts[supercategory].union_with(concept), concept_colors[supercategory])
                else:
                    concept_colors.update({supercategory:next(colors)})
                    space.add_concept(supercategory, concept, concept_colors[supercategory])"""
                #print('added ', example)
                #print(concepts[example]['supercategory'])
        #except Exception:
        #    print(example,'is inconsistent')


def one_shot(point, supercategory='none'):
    #todo find lowest concept
    '''print(point)'''
    for concept in space._concepts:
        '''print(concept)
        print('membership:')
        print(space._concepts[concept].membership_of(point))
        print()'''
    if supercategory=='none':
        superkey = max(space._concepts,key= lambda candidate:space._concepts[candidate].membership_of(point))
    else:
        superkey = supercategory
    
    supercategory = space._concepts[superkey]
    avg = lambda values: sum(values)/len(values) if len(values) else float('inf')
    #print([[cuboid._p_max[dim]-cuboid._p_min[dim] for cuboid in supercategory._core._cuboids if not cuboid._p_max[dim]==float('inf')]for dim in range(space._n_dim)])
    avg_sizes=[avg([cuboid._p_max[dim]-cuboid._p_min[dim] for cuboid in supercategory._core._cuboids if not cuboid._p_max[dim]==float('inf')]) for dim in range(space._n_dim)]
    print('supercategory is:')
    print(superkey)
    print(avg_sizes)
    print()
    p_min = [point[i]-1/2*avg_sizes[i] for i in range(space._n_dim)]
    p_max = [point[i]+1/2*avg_sizes[i] for i in range(space._n_dim)]
    cuboid=Cuboid(p_min, p_max, space._domains)#not working in bigger examples
    core=Core([cuboid],space._domains)
    weights=Weights(space._def_dom_weights,space._def_dim_weights)
    concept=Concept(core, 1.0, 0.5, weights)
    return concept
        
''

#%matplotlib widget
concepts = xml_to_dict(path)
concepts = xml_to_dict(path2, concepts)

#print(concepts)

domains = {}

for concept in concepts.values():
    domains.update(concept)
del(domains['supercategory'])


example='bear'
        
to_learn={example:concepts[example]}
del(concepts[example])
mammal={key:concepts[key] for key in concepts if concepts[key]['supercategory']=='mammal'}

to_del=[]
for concept in concepts:
    if concepts[concept]['supercategory']=='object' or concepts[concept]['supercategory']=='amphibian':
        to_del.append(concept)
for conc in to_del:
    del(concepts[conc])
        
concepts_into_space(concepts.copy(), except_for=to_learn[example]['supercategory'])

#print(space._concepts['reptile'])

dimension_values=[]
concepts=to_learn
consistent = True
#print(bear)
for domain in space._domains.keys():
    dimension_values.append([list(concepts[example].get(domain,{str(key):float("-inf") for key in range(len(space._domains[domain]))}).values())
                         ,list(concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values())])
    if(not(len(dimension_values[-1][0])==len(space._domains[domain]))):
        print(domain+' in '+example+' is inconsistent')
        consistent=False
        break

domains = {}
for key in concepts[example]:
    if(not(key=='supercategory')):
        domains[key]=space._domains[key]

'''print(dimension_values)
print(domains)'''

'''if(consistent):
        p_min=[value for domain in dimension_values for value in domain[0]]
        p_max=[value for domain in dimension_values for value in domain[1]]

        #try:
        c_example = Cuboid(p_min, p_max, domains)
        s_example = Core([c_example], domains)
        w_example = Weights(space._def_dom_weights,space._def_dim_weights)
        bear = Concept(s_example, 1.0, 0.5, w_example)
        space.add_concept(example, bear, 'yellow')'''

'''for concept in space._concepts:'''
#print(concept)
p_max=[value for domain in dimension_values for value in domain[1]]
print('bear:')
print(p_max)

space.add_concept(example+' from membership',one_shot(p_max),'forestgreen')
space.add_concept(example+' from animal',one_shot(p_max,supercategory='animal'),'lime')

example='camel'

to_learn={example:mammal[example]}
dimension_values=[]
concepts=to_learn
consistent = True
#print(bear)
for domain in space._domains.keys():
    dimension_values.append([list(concepts[example].get(domain,{str(key):float("-inf") for key in range(len(space._domains[domain]))}).values())
                         ,list(concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values())])
    if(not(len(dimension_values[-1][0])==len(space._domains[domain]))):
        print(domain+' in '+example+' is inconsistent')
        consistent=False
        break

domains = {}
for key in concepts[example]:
    if(not(key=='supercategory')):
        domains[key]=space._domains[key]
        
p_max=[value for domain in dimension_values for value in domain[1]]        
space.add_concept(example+' from membership',one_shot(p_max),'yellow')
space.add_concept(example+' from animal',one_shot(p_max,supercategory='animal'),'gold')

concepts=mammal
ordered_dict={}
for example in concepts:
    if concepts[example]['supercategory'] in ordered_dict:
        ordered_dict[concepts[example]['supercategory']].append(example)
    else:
        ordered_dict[concepts[example]['supercategory']]=[example]
for category in ordered_dict:
    print('putting',category,'into space')
    cuboids=[]
    domains = {}
    for example in ordered_dict[category]:
        if not is_consistent(concepts[example]):
            print(example,' is inconsistent')
        else:
            subdomains={domain:space._domains[domain] for domain in concepts[example] if not domain == 'supercategory'}
            point=[concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values() for domain in space._domains]
            point=[p for dom in point for p in dom]
            #print(example)
            #print('point: ',point)
            cuboids.append(Cuboid([dim if not dim==float('inf') else -dim for dim in point], point, subdomains))
            domains.update(subdomains)
            #print(cuboids[-1]._p_max)
    #for cuboid in cuboids:
        #print('cuboid:',cuboid)
    core=from_cuboids(cuboids,domains)
    weights=Weights(space._def_dom_weights,space._def_dim_weights)
    concept=Concept(core, 1.0, 0.5, weights)
    space.add_concept(category,concept,'orange')
    
    
        #print(space._concepts[concept].membership_of([p if not p==float('-inf') else 0 for p in p_min]))
        #print(space._concepts[concept].membership_of(p_max))
            #print(bear.subset_of(space._concepts[concept]))
'''print('amphibian:')
print(space._concepts['amphibian'])'''
ci.init()