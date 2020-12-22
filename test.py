import xml.etree.ElementTree as ET
import os
import sys
sys.path.append("ConceptualSpaces-master\\conceptual_spaces")
import cs.cs as space
from cs.weights import Weights
from cs.cuboid import Cuboid
from cs.core import Core
from cs.concept import Concept
import visualization.concept_inspector as ci



path = "Thesis\\Datasets\\Dual-PECCS\\s1s2_deploy_test\\files\\knowledge_base\\prototypes\\prototypes.xml"
path2 = "Thesis\\Datasets\\Dual-PECCS\\s1s2_deploy_test\\files\\knowledge_base\\exemplars\\exemplars.xml"


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
        domain_dict['supercategory']=xml_elem.find('family').text
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
            if(domain_name=='hasPart'):
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

    
def concepts_into_space(concepts, domains={}):
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
    
    
    for example in concepts:
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
                    print(domain+' in '+example+' is inconsistent')
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
                concept = Concept(s_example, 1.0, 24.0, w_example)

                supercategory = concepts[example]['supercategory']
                if(supercategory in space._concepts):
                    space.add_concept(supercategory, space._concepts[supercategory].union_with(concept))
                else:
                    space.add_concept(supercategory, concept)
                print('added ', example)
                print(concepts[example]['supercategory'])


concepts = xml_to_dict(path)
concepts = xml_to_dict(path2, concepts)

domains = {}

for concept in concepts.values():
    domains.update(concept)
del(domains['supercategory'])


        
bear={'Rhinoceros':concepts['Rhinoceros']}
del(concepts['Rhinoceros'])
concepts_into_space(concepts)

dimension_values=[]
concepts=bear
example='Rhinoceros'
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

print(dimension_values)
print(domains)

if(consistent):
        p_min=[value for domain in dimension_values for value in domain[0]]
        p_max=[value for domain in dimension_values for value in domain[1]]

        #try:
        c_example = Cuboid(p_min, p_max, domains)
        s_example = Core([c_example], domains)
        w_example = Weights(space._def_dom_weights,space._def_dim_weights)
        bear = Concept(s_example, 1.0, 0.5, w_example)
        space.add_concept('Rhino', bear, 'yellow')

        for concept in space._concepts:
            print(concept)
            #print(space._concepts[concept].membership_of([p if not p==float('-inf') else 0 for p in p_min]))
            print(space._concepts[concept].membership_of(p_max))
            #print(bear.subset_of(space._concepts[concept]))
ci.init(dims=[12,13,10])
