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
import numpy as np
from matplotlib import patches


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
                

def concepts_into_space(concepts, domains={}, except_for=None, colors=['b','g','r','c','m','brown','rosybrown','darkslategrey','pink','grey'], superc='animal'):
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
    
    if space._n_dim==None:
        space.init(len(dimension_names),domain_mapping,dimension_names)
    else:
        print('ne')
    
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
        
    if not superc==None:
        animal_concept=[]
        for concept in space._concepts.values():
            animal_concept+=concept._core._cuboids
        #print('animal cubs:\n',animal_concept)
        core=from_cuboids(animal_concept,space._domains)
        weights=Weights(space._def_dom_weights,space._def_dim_weights)
        concept=Concept(core, 1.0, 0.5, weights)
        space.add_concept(superc,concept,next(colors))
        
        
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

def point_to_concept(point, name):

    domains = domain_from_points(point)
    p_min=[value if not value==float('inf') else float('-inf') for value in point]
    c_example = Cuboid(p_min, point, domains)
    s_example = Core([c_example], domains)
    w_example = Weights(space._def_dom_weights,space._def_dim_weights)
    concept = Concept(s_example, 1.0, 0.5, w_example)
    space.add_concept(concept, name)

def one_shot(point, supercategory='none', mechanism=None, sibblings=None, name="newlearned"):
    #todo find lowest concept

    if supercategory=='none':
        superkey = max(space._concepts,key= lambda candidate:space._concepts[candidate].membership_of(point))
    else:
        superkey = supercategory
    
    if(mechanism==None):
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
    else:
        space.add_concept(name,mechanism(point, supercategory, sibblings),color='y')

def domains_from_point(point):
    domains={}
    for domain in space._domains:
        if all(not point[dim]==float('inf') for dim in space._domains[domain]):
            domains[domain]=space._domains[domain]
    return domains
    
    
def get_sibblings(supercat):
    sibblings=[]
    for concept in space._concepts:
        if not concept==supercat:
            if space._concepts[concept].subset_of(space._concepts[supercat])==1.0:
                sibblings.append(concept)
    return sibblings
        
def two_cuboids(point, supercategory, sibblings=None):
    #fig,ax = plt.subplots(1)
    
    n_dims=len(point)
    if sibblings==None:
        sibblings=get_sibblings(supercategory)
        
    variances=[]
    correlations=[]
    stds=[]#np.std(centers)
    avg = np.average#lambda values: sum(values)/len(values)
    for sibbling in sibblings:
        centers = np.array([[p_max-p_min for p_max,p_min in zip(cuboid._p_max, cuboid._p_min)] for cuboid in space._concepts[sibbling]._core._cuboids]).T
        means=[]
        for dimension in range(len(centers)):
            means.append(avg([value for value in centers[dimension] if not value==float('inf')]))
            for value in range(len(centers[dimension])):
                if centers[dimension,value]==float('inf'):
                    centers[dimension,value]=means[-1]
        if(len(centers.T)==1):
            centers=np.array([[x[0],x[0]] for x in centers])
        stds.append(np.std(centers,1))
        #covmat = np.cov(centers)
        corr=np.corrcoef(centers)
        #plt.scatter(centers[0],centers[1])
        #variances.append(covmat)
        if not np.isnan(corr[0,0]):
            correlations.append(corr)
            
    meanstds=np.zeros(n_dims)
    meancorrelations=np.zeros([n_dims, n_dims])
    for i in range(n_dims):
        meanstds[i] = avg([stds[sib][i] for sib in range(len(stds))])
        #meanvariances[i] = avg([(variance_of_sibbling[i,i])**(1/2) for variance_of_sibbling in variances])
        for j in range(n_dims):
            meancorrelations[i,j]= avg([correlations_of_sibbling[i,j] for correlations_of_sibbling in correlations])
    #print(meanvariances)
    print(meanstds)
    #print([avg([stds[sib][dim] for sib in range(len(stds))]) for dim in range(n_dims)])
    #print(meancorrelations)
    maindim=0
    cub1=np.zeros(n_dims)
    cub2=np.zeros(n_dims)
    cub1[maindim]=point[maindim]-meanstds[maindim]
    cub2[maindim]=point[maindim]+meanstds[maindim]
    
    for dim in range(n_dims):
        if not dim==maindim:
            cub1[dim]=point[dim]-meancorrelations[maindim,dim]*meanstds[maindim]
            cub2[dim]=point[dim]+meancorrelations[maindim,dim]*meanstds[maindim]
        
    c1_p_min=[2*cub1[dim]-point[dim] if cub1[dim] < point[dim] else point[dim] for dim in range(n_dims)]
    c1_p_max=[2*cub1[dim]-point[dim] if cub1[dim] > point[dim] else point[dim] for dim in range(n_dims)]
    
    c2_p_min=[2*cub2[dim]-point[dim] if cub2[dim] < point[dim] else point[dim] for dim in range(n_dims)]
    c2_p_max=[2*cub2[dim]-point[dim] if cub2[dim] > point[dim] else point[dim] for dim in range(n_dims)]
    

    '''rect=patches.Rectangle(c1_p_min[0:2],c1_p_max[0]-c1_p_min[0],c1_p_max[1]-c1_p_min[1])
    ax.add_patch(rect)
    rect=patches.Rectangle(c2_p_min[0:2],c2_p_max[0]-c2_p_min[0],c2_p_max[1]-c2_p_min[1])
    ax.add_patch(rect)
    print(rect.get_xy())
    plt.show()'''
    
    domains = domains_from_point(c1_p_min)
    core=Core([Cuboid(c1_p_min, c1_p_max, domains),Cuboid(c2_p_min,c2_p_max,domains)],domains)
    weights=Weights(space._def_dom_weights,space._def_dim_weights)
    concept=Concept(core, 1.0, 0.5, weights)
    return concept
    
    
''

def generate(example):
    #parameters

    #read concepts from prototypes and exemplars
    concepts = xml_to_dict(path)
    concepts = xml_to_dict(path2, concepts)

    #dict domains lists all domains shich later will be in cs
    domains = {}
    for concept in concepts.values():
        domains.update(concept)
    del(domains['supercategory'])

    #remove the testexample from traindata
    #del(concepts[example])
    target=concepts[example]['supercategory']
    #mammal={key:concepts[key] for key in concepts if concepts[key]['supercategory']=='mammal'}


    #remove objects for now (into_space still only animals)
    to_del=[]
    for concept in concepts:
        #amphibian=error, whale=too big
        if concepts[concept]['supercategory']=='object' or concepts[concept]['supercategory']=='amphibian' or concept=='whale':
            to_del.append(concept)
    for conc in to_del:
        del(concepts[conc])

    #testset = everythink that makes up superategory of example except for example
    trainset={}
    testset={}
    for concept in concepts:
        if not concepts[concept]['supercategory']==target:
            trainset[concept]=concepts[concept]
        elif not concept==example:
            testset[concept]=concepts[concept]


    #the supercategory of to_learn will not be put into space
    concepts_into_space(trainset)

    #get point from dict of example
    dimension_values=[]
    consistent = True
    for domain in space._domains.keys():
        dimension_values.append([list(concepts[example].get(domain,{str(key):float("-inf") for key in range(len(space._domains[domain]))}).values())
                             ,list(concepts[example].get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values())])
        if(not(len(dimension_values[-1][0])==len(space._domains[domain]))):
            print(domain+' in '+example+' is inconsistent')
            consistent=False
            break

    p_max=[value for domain in dimension_values for value in domain[1]]

    '''domains = {}
    for key in concepts[example]:
        if(not(key=='supercategory')):
            domains[key]=space._domains[key]'''


    #save time
    sibblings = [family for family in ['mammal', 'insect', 'arthropod', 'reptile', 'fish', 'arachnid', 'bird'] if not family==target] 
    #oneshot
    try:
        sibblings
    except NameError:
        sibblings=get_sibblings('animal')
    one_shot(p_max, space._concepts['animal'],two_cuboids,sibblings=sibblings,name=target+'from'+example)



    target_dict={}

    concepts_into_space(testset, superc=None, colors=['orange'])

    ci.init()