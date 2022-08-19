import pytest
import numpy as np
from rakun2 import RakunKeyphraseDetector

example_document = """
Marsupial
From Wikipedia, the free encyclopedia
Jump to navigation
Jump to search
This article is about the mammals. For frogs, see Marsupial frog.
Marsupials
Temporal range: Paleocene–Recent
PreꞒ
Ꞓ
O
S
D
C
P
T
J
K
Pg
N
Possible Late Cretaceous records
Marsupialia.jpg
Clockwise from left: eastern grey kangaroo, Virginia opossum, long-nosed bandicoot, Monito del monte and Tasmanian devil representing the orders Diprotodontia, Didelphimorphia, Peramelemorphia, Microbiotheria and Dasyuromorphia respectively
Scientific classification e
Kingdom: 	Animalia
Phylum: 	Chordata
Class: 	Mammalia
Clade: 	Marsupialiformes
Infraclass: 	Marsupialia
Illiger, 1811
Orders

    Didelphimorphia
    Paucituberculata
    Australidelphia
        Microbiotheria
        Dasyuromorphia
        Peramelemorphia
        Notoryctemorphia
        Diprotodontia
        †Yalkaparidontia
        †Polydolopimorphia?

Marsupial biogeography present day - dymaxion map.png
Present-day distribution of marsupials (blue; excludes introduced presence in New Zealand)

Marsupials are any members of the mammalian infraclass Marsupialia. All extant marsupials are endemic to Australasia, Wallacea and the Americas. A distinctive characteristic common to most of these species is that the young are carried in a pouch. Marsupials include opossums, Tasmanian devils, kangaroos, koalas, wombats, wallabies, bandicoots, and the extinct thylacine.

Marsupials represent the clade originating from the last common ancestor of extant metatherians, the group containing all mammals more closely related to marsupials than to placentals. They give birth to relatively undeveloped young that often reside in a pouch located on their mothers' abdomen for a certain amount of time. Close to 70% of the 334 extant species occur on the Australian continent (the mainland, Tasmania, New Guinea and nearby islands). The remaining 30% are found in the Americas—primarily in South America, thirteen in Central America, and one species, the Virginia opossum, in North America, north of Mexico.

The word marsupial comes from marsupium, the technical term for the abdominal pouch. It, in turn, is borrowed from Latin and ultimately from the ancient Greek μάρσιππος mársippos, meaning "pouch".
Contents

    1 Taxonomy
        1.1 Classification
        1.2 Phylogenetic relationships
    2 Anatomy
        2.1 Skull and teeth
        2.2 Torso
        2.3 General and convergences
        2.4 Body temperature
        2.5 Reproductive system
    3 Geography
    4 Interaction with Europeans
    5 Evolutionary history
    6 See also
    7 Notes
    8 References
    9 Further reading
    10 External links

Taxonomy

Marsupials are taxonomically identified as members of mammalian infraclass Marsupialia, first described as a family under the order Pollicata by German zoologist Johann Karl Wilhelm Illiger in his 1811 work Prodromus Systematis Mammalium et Avium. However, James Rennie, author of The Natural History of Monkeys, Opossums and Lemurs (1838), pointed out that the placement of five different groups of mammals – monkeys, lemurs, tarsiers, aye-ayes and marsupials (with the exception of kangaroos, that were placed under the order Salientia) – under a single order (Pollicata) did not appear to have a strong justification. In 1816, French zoologist George Cuvier classified all marsupials under the order Marsupialia.[1][2] In 1997, researcher J. A. W. Kirsch and others accorded infraclass rank to Marsupialia.[2] There are two primary divisions: American marsupials (Ameridelphia) and Australian marsupials (Australidelphia) of which one, the monito del monte, is actually native to South America.[3]
Classification

Marsupialia is further divided as follows:[3]

† – Extinct

    Superorder Ameridelphia
        Order Didelphimorphia (127 species)
            Family Didelphidae: opossums
        Order Paucituberculata (seven species)
            Family Caenolestidae: shrew opossums
    Superorder Australidelphia
        Order Microbiotheria (three species)
            Family Microbiotheriidae: monitos del monte
        Order †Yalkaparidontia (incertae sedis)
        Order Dasyuromorphia (75 species)
            Family †Thylacinidae: thylacine
            Family Dasyuridae: antechinuses, quolls, dunnarts, Tasmanian devil, and relatives
            Family Myrmecobiidae: numbat
        Order Notoryctemorphia (two species)
            Family Notoryctidae: marsupial moles
        Order Peramelemorphia (24 species)
            Family Thylacomyidae: bilbies
            Family †Chaeropodidae: pig-footed bandicoots
            Family Peramelidae: bandicoots and allies
        Order Diprotodontia (137 species)
            Suborder Vombatiformes
                Family Vombatidae: wombats
                Family Phascolarctidae: koalas
                Family † Diprotodontidae: giant wombats
                Family † Palorchestidae: marsupial tapirs
                Family † Thylacoleonidae: marsupial lions
            Suborder Phalangeriformes
                Family Acrobatidae: feathertail glider and feather-tailed possum
                Family Burramyidae: pygmy possums
                Family †Ektopodontidae: sprite possums
                Family Petauridae: striped possum, Leadbeater's possum, yellow-bellied glider, sugar glider, mahogany glider, squirrel glider
                Family Phalangeridae: brushtail possums and cuscuses
                Family Pseudocheiridae: ringtailed possums and relatives
                Family Tarsipedidae: honey possum
            Suborder Macropodiformes
                Family Macropodidae: kangaroos, wallabies, and relatives
                Family Potoroidae: potoroos, rat kangaroos, bettongs
                Family Hypsiprymnodontidae: musky rat-kangaroo
                Family † Balbaridae: basal quadrupedal kangaroos

Phylogenetic relationships

Comprising over 300 extant species, several attempts have been made to accurately interpret the phylogenetic relationships among the different marsupial orders. Studies differ on whether Didelphimorphia or Paucituberculata is the sister group to all other marsupials.[4] Though the order Microbiotheria (which has only one species, the monito del monte) is found in South America, morphological similarities suggest it is closely related to Australian marsupials.[5] Molecular analyses in 2010 and 2011 identified Microbiotheria as the sister group to all Australian marsupials. However, the relations among the four Australidelphid orders are not as well understood. The cladogram below, depicting the relationships among the various marsupial orders, is based on a 2015 phylogenetic study.[4]
Marsupialia 	
  	

DidelphimorphiaA hand-book to the marsupialia and monotremata (Plate XXXII) (white background).jpg
 
  	
  	

PaucituberculataPhylogenetic tree of marsupials derived from retroposon data (Paucituberculata).png
 
Australidelphia 	
  	

Microbiotheria
 
  	
  	

DiprotodontiaA monograph of the Macropodidæ, or family of kangaroos (9398404841) white background.jpg
 
  	
  	

NotoryctemorphiaPhylogenetic tree of marsupials derived from retroposon data (Notoryctemorphia).png
 
  	
  	

DasyuromorphiaPhylogenetic tree of marsupials derived from retroposon data (Dasyuromorphia).png 
 
  	

PeramelemorphiaPhylogenetic tree of marsupials derived from retroposon data (Paramelemorphia).png
 
 
 
 
 
 
	New World marsupials













Australasian marsupials
 

DNA evidence supports a South American origin for marsupials, with Australian marsupials arising from a single Gondwanan migration of marsupials from South America, across Antarctica, to Australia.[6][7] There are many small arboreal species in each group. The term "opossum" is used to refer to American species (though "possum" is a common abbreviation), while similar Australian species are properly called "possums".
Anatomy
Koala
(Phascolarctos cinereus)

Marsupials have the typical characteristics of mammals—e.g., mammary glands, three middle ear bones, and true hair. There are, however, striking differences as well as a number of anatomical features that separate them from eutherians.

In addition to the front pouch, which contains multiple teats for the sustenance of their young, marsupials have other common structural features. Ossified patellae are absent in most modern marsupials (though a small number of exceptions are reported)[8] and epipubic bones are present. Marsupials (and monotremes) also lack a gross communication (corpus callosum) between the right and left brain hemispheres.[9]
Skull and teeth

The skull has peculiarities in comparison to placental mammals. In general, the skull is relatively small and tight. Holes (foramen lacrimale) are located in the front of the orbit. The cheekbone is enlarged and extends farther to the rear. The angular extension (processus angularis) of the lower jaw is bent toward the center. Another feature is the hard palate which, in contrast to the placental mammals' foramina, always have more openings. The teeth differ from that of placental mammals, so that all taxa except wombats have a different number of incisors in the upper and lower jaws. The early marsupials had a dental formula from 5.1.3.44.1.3.4, that is, per quadrant; they have five (maxillary) or four (mandibular) incisors, one canine, three premolars and four molars, for a total of 50 teeth. Some taxa, such as the opossum, have the original number of teeth. In other groups the number of teeth is reduced. The dental formula for Macropodidae (kangaroos and wallabies etc.) is 3/1 – (0 or 1)/0 – 2/2 – 4/4. Marsupials in many cases have 40 to 50 teeth, significantly more than placental mammals. The second set of teeth grows in only at the 3rd premolar site and back; all teeth more anterior to that erupt initially as permanent teeth.
Torso

Few general characteristics describe their skeleton. In addition to unique details in the construction of the ankle, epipubic bones (ossa epubica) are observed projecting forward from the pubic bone of the pelvis. Since these are present in males and pouchless species, it is believed that they originally had nothing to do with reproduction, but served in the muscular approach to the movement of the hind limbs. This could be explained by an original feature of mammals, as these epipubic bones are also found in monotremes. Marsupial reproductive organs differ from the placental mammals. For them, the reproductive tract is doubled. The females have two uteri and two vaginas, and before birth, a birth canal forms between them, the median vagina.[9] The males have a split or double penis lying in front of the scrotum.[10]

A pouch is present in most, but not all, species. Many marsupials have a permanent bag, whereas in others the pouch develops during gestation, as with the shrew opossum, where the young are hidden only by skin folds or in the fur of the mother. The arrangement of the pouch is variable to allow the offspring to receive maximum protection. Locomotive kangaroos have a pouch opening at the front, while many others that walk or climb on all fours have the opening in the back. Usually, only females have a pouch, but the male water opossum has a pouch that is used to accommodate his genitalia while swimming or running.
General and convergences
The sugar glider, a marsupial, (left) and flying squirrel, a rodent, (right) are examples of convergent evolution.

Marsupials have adapted to many habitats, reflected in the wide variety in their build. The largest living marsupial, the red kangaroo, grows up to 1.8 metres (5 ft 11 in) in height and 90 kilograms (200 lb) in weight, but extinct genera, such as Diprotodon, were significantly larger and heavier. The smallest members of this group are the marsupial mice, which often reach only 5 centimetres (2.0 in) in body length.

Some species resemble placental mammals and are examples of convergent evolution. This convergence is evident in both brain evolution[11] and behaviour.[12] The extinct Thylacine strongly resembled the placental wolf, hence one of its nicknames "Tasmanian wolf". The ability to glide evolved in both marsupials (as with sugar gliders) and some placental mammals (as with flying squirrels), which developed independently. Other groups such as the kangaroo, however, do not have clear placental counterparts, though they share similarities in lifestyle and ecological niches with ruminants.
Body temperature

Marsupials, along with monotremes (platypuses and echidnas), typically have lower body temperatures than similarly-sized placental mammals (eutherians).[13]
Reproductive system
See also: Kangaroo § Reproduction and life cycle
Female eastern grey kangaroo with a joey in her pouch

Marsupials' reproductive systems differ markedly from those of placental mammals.[14][15] During embryonic development, a choriovitelline placenta forms in all marsupials. In bandicoots, an additional chorioallantoic placenta forms, although it lacks the chorionic villi found in eutherian placentas.

The evolution of reproduction in marsupials, and speculation about the ancestral state of mammalian reproduction, have engaged discussion since the end of the 19th century. Both sexes possess a cloaca,[15] which is connected to a urogenital sac used to store waste before expulsion. The bladder of marsupials functions as a site to concentrate urine and empties into the common urogenital sinus in both females and males.[15]

Male reproductive system
Reproductive tract of a male macropod

Most male marsupials, except for macropods[16] and marsupial moles,[17] have a bifurcated penis, separated into two columns, so that the penis has two ends corresponding to the females' two vaginas.[9][15][18][19][10][20][21] The penis is used only during copulation, and is separate from the urinary tract.[10][15] It curves forward when erect,[22] and when not erect, it is retracted into the body in an S-shaped curve.[10] Neither marsupials nor monotremes possess a baculum.[9] The shape of the glans penis varies among marsupial species.[10][23][24][25]

The male thylacine had a pouch that acts as a protective sheath, covering his external reproductive organs while running through thick brush.[26]

The shape of the urethral grooves of the males' genitalia is used to distinguish between Monodelphis brevicaudata, Monodelphis domestica, and Monodelphis americana. The grooves form 2 separate channels that form the ventral and dorsal folds of the erectile tissue.[27] Several species of dasyurid marsupials can also be distinguished by their penis morphology.[28] The only accessory sex glands marsupials possess are the prostate and bulbourethral glands.[29] Male marsupials have 1-3 pairs of bulbourethral glands.[30] There are no ampullae, seminal vesicles or coagulating glands.[31][18] The prostate is proportionally larger in marsupials than in placental mammals.[10] During the breeding season, the male tammar wallaby's prostate and bulbourethral gland enlarge. However, there does not appear to be any seasonal difference in the weight of the testes.[32]
Female reproductive system
See also: Birth § Marsupials
Female reproductive anatomy of several marsupial species

Female marsupials have two lateral vaginas, which lead to separate uteri, but both open externally through the same orifice. A third canal, the median vagina, is used for birth. This canal can be transitory or permanent.[9] Some marsupial species are able to store sperm in the oviduct after mating.[33]

Marsupials give birth at a very early stage of development; after birth, newborn marsupials crawl up the bodies of their mothers and attach themselves to a teat, which is located on the underside of the mother, either inside a pouch called the marsupium, or open to the environment. Mothers often lick their fur to leave a trail of scent for the newborn to follow to increase chances of making it into the marsupium. There they remain for a number of weeks, attached to the teat. The offspring are eventually able to leave the marsupium for short periods, returning to it for warmth, protection, and nourishment.
Early development
	
This section may contain content that is repetitive or redundant of text elsewhere in the article. Please help improve it by merging similar text or removing repeated statements. (November 2017)
Child holding rescued agile wallaby joey. Cooktown. 2008

Prenatal development differs between marsupials and placental mammals. Key aspects of the first stages of placental mammal embryo development, such as the inner cell mass and the process of compaction, are not found in marsupials.[34] The cleavage stages of marsupial development are very variable between groups and aspects of marsupial early development are not yet fully understood.

An early birth removes a developing marsupial from its mother's body much sooner than in placental mammals; thus marsupials have not developed a complex placenta to protect the embryo from its mother's immune system. Though early birth puts the tiny newborn marsupial at a greater environmental risk, it significantly reduces the dangers associated with long pregnancies, as there is no need to carry a large fetus to full term in bad seasons. Marsupials are extremely altricial animals, needing to be intensely cared for immediately following birth (cf. precocial).

Because newborn marsupials must climb up to their mother's teats, their front limbs and facial structures are much more developed than the rest of their bodies at the time of birth.[35][36] This requirement has been argued to have resulted in the limited range of locomotor adaptations in marsupials compared to placentals. Marsupials must develop grasping forepaws during their early youth, making the evolutive transition from these limbs into hooves, wings, or flippers, as some groups of placental mammals have done, more difficult. However, several marsupials do possess atypical forelimb morphologies, such as the hooved forelimbs of the pig-footed bandicoot, suggesting that the range of forelimb specialization is not as limited as assumed.[37]

An infant marsupial is known as a joey. Marsupials have a very short gestation period—usually around four to five weeks, but as low as 12 days for some species—and the joey is born in an essentially fetal state. The blind, furless, miniature newborn, the size of a jelly bean,[38][failed verification] crawls across its mother's fur to make its way into the pouch, where it latches onto a teat for food. It will not re-emerge for several months, during which time it develops fully. After this period, the joey begins to spend increasing lengths of time out of the pouch, feeding and learning survival skills. However, it returns to the pouch to sleep, and if danger threatens, it will seek refuge in its mother's pouch for safety.

Joeys stay in the pouch for up to a year in some species, or until the next joey is born. A marsupial joey is unable to regulate its own body temperature and relies upon an external heat source. Until the joey is well furred and old enough to leave the pouch, a pouch temperature of 30–32 °C (86–90 °F) must be constantly maintained.

Joeys are born with "oral shields". In species without pouches or with rudimentary pouches these are more developed than in forms with well-developed pouches, implying a role in maintaining the young attached to the mother's teat.[39]
Geography

In Australasia, marsupials are found in Australia, Tasmania and New Guinea; throughout the Maluku Islands, Timor and Sulawesi to the west of New Guinea, and in the Bismarck Archipelago (including the Admiralty Islands) and Solomon Islands to the east of New Guinea.

In America, marsupials are found throughout South America, excluding the central/southern Andes and parts of Patagonia; and through Central America and south-central Mexico, with a single species widespread in the eastern United States and along the Pacific coast.
Interaction with Europeans

The first American marsupial (and marsupial in general) that a European encountered was the common opossum. Vicente Yáñez Pinzón, commander of the Niña on Christopher Columbus' first voyage in the late fifteenth century, collected a female opossum with young in her pouch off the South American coast. He presented them to the Spanish monarchs, though by then the young were lost and the female had died. The animal was noted for its strange pouch or "second belly", and how the offspring reached the pouch was a mystery.[40][41]

On the other hand, it was the Portuguese who first described Australasian marsupials. António Galvão, a Portuguese administrator in Ternate (1536–40), wrote a detailed account of the northern common cuscus (Phalanger orientalis):[40]

    Some animals resemble ferrets, only a little bigger. They are called Kusus. They have a long tail with which they hang from the trees in which they live continuously, winding it once or twice around a branch. On their belly they have a pocket like an intermediate balcony; as soon as they give birth to a young one, they grow it inside there at a teat until it does not need nursing anymore. As soon as she has borne and nourished it, the mother becomes pregnant again.

From the start of the 17th century more accounts of marsupials arrived. For instance, a 1606 record of an animal, killed on the southern coast of New Guinea, described it as "in the shape of a dog, smaller than a greyhound", with a snakelike "bare scaly tail" and hanging testicles. The meat tasted like venison, and the stomach contained ginger leaves. This description appears to closely resemble the dusky pademelon (Thylogale brunii), in which case this would be the earliest European record of a member of the kangaroo family (Macropodidae).[42][40]
Evolutionary history
See also: Metatheria, Evolution of Macropodidae, and Evolution of mammals
Isolated petrosals of Djarthia murgonensis, Australia's oldest marsupial fossils[43]
Dentition of the herbivorous eastern grey kangaroo, as illustrated in Knight's Sketches in Natural History

The relationships among the three extant divisions of mammals (monotremes, marsupials, and placentals) were long a matter of debate among taxonomists.[44] Most morphological evidence comparing traits such as number and arrangement of teeth and structure of the reproductive and waste elimination systems as well as most genetic and molecular evidence favors a closer evolutionary relationship between the marsupials and placental mammals than either has with the monotremes.[45]
Phylogenetic tree of marsupials derived from retroposon data[7]

The ancestors of marsupials, part of a larger group called metatherians, probably split from those of placental mammals (eutherians) during the mid-Jurassic period, though no fossil evidence of metatherians themselves are known from this time.[46] From DNA and protein analyses, the time of divergence of the two lineages has been estimated to be around 100 to 120 mya.[40] Fossil metatherians are distinguished from eutherians by the form of their teeth; metatherians possess four pairs of molar teeth in each jaw, whereas eutherian mammals (including true placentals) never have more than three pairs.[47] Using this criterion, the earliest known metatherian was thought to be Sinodelphys szalayi, which lived in China around 125 mya.[48][49][50] However Sinodelphys was later reinterpreted as an early member of Eutheria. The unequivocal oldest known metatherians are now 110 million years old fossils from western North America.[51] Metatherians were widespread in North America and Asia during the Late Cretaceous, but suffered a severe decline during the end-Cretaceous extinction event.[52]

Marsupials spread to South America from North America during the Paleocene, possibly via the Aves Ridge.[53][54][55] Northern Hemisphere metatherians, which were of low morphological and species diversity compared to contemporary placental mammals, eventually became extinct during the Miocene epoch.[56]

Cladogram from Wilson et al. (2016)[57]
Metatheria 	
  	

Holoclemensia
 
  	
  	
  	

Pappotherium
 
  	
  	

Sulestes
 
  	

Oklatheridium
 
  	
  	

Tsagandelta
 
  	
  	

Lotheridium
 
  	
  	

Deltatheroides
 
  	

Deltatheridium
 
  	
  	

Nanocuris
 
  	

Atokatheridium
 
 
 
 
 
 
 
Marsupialiformes 	
  	
  	

Gurlin Tsav skull
 
  	
  	

Borhyaenidae
 
  	
  	

Mayulestes
 
  	
  	

Jaskhadelphys
 
  	
  	

Andinodelphys
 
  	

Pucadelphys
 
 
 
 
 
 
  	
  	

Asiatherium
 
  	
  	
  	

Iugomortiferum
 
  	

Kokopellia
 
 
  	

Aenigmadelphys
 
  	

Anchistodelphys
 
  	
Glasbiidae 	

Glasbius
 
Pediomyidae 	

Pediomys
 
 
Stagodontidae 	
  	

Pariadens
 
  	
  	

Eodelphis
 
  	

Didelphodon
 
 
 
Alphadontidae 	
  	

Turgidodon
 
  	

Alphadon
 
  	

Albertatherium
 
 
  	

Marsupialia
 
 
 
 
 
 

In South America, the opossums evolved and developed a strong presence, and the Paleogene also saw the evolution of shrew opossums (Paucituberculata) alongside non-marsupial metatherian predators such as the borhyaenids and the saber-toothed Thylacosmilus. South American niches for mammalian carnivores were dominated by these marsupial and sparassodont metatherians, which seem to have competitively excluded South American placentals from evolving carnivory.[58] While placental predators were absent, the metatherians did have to contend with avian (terror bird) and terrestrial crocodylomorph competition. Marsupials were excluded in turn from large herbivore niches in South America by the presence of native placental ungulates (now extinct) and xenarthrans (whose largest forms are also extinct). South America and Antarctica remained connected until 35 mya, as shown by the unique fossils found there. North and South America were disconnected until about three million years ago, when the Isthmus of Panama formed. This led to the Great American Interchange. Sparassodonts disappeared for unclear reasons – again, this has classically assumed as competition from carnivoran placentals, but the last sparassodonts co-existed with a few small carnivorans like procyonids and canines, and disappeared long before the arrival of macropredatory forms like felines,[59] while didelphimorphs (opossums) invaded Central America, with the Virginia opossum reaching as far north as Canada.

Marsupials reached Australia via Antarctica during the Early Eocene, around 50 mya, shortly after Australia had split off.[n 1][n 2] This suggests a single dispersion event of just one species, most likely a relative to South America's monito del monte (a microbiothere, the only New World australidelphian). This progenitor may have rafted across the widening, but still narrow, gap between Australia and Antarctica. The journey must not have been easy; South American ungulate[63][64][65] and xenarthran[66] remains have been found in Antarctica, but these groups did not reach Australia.

In Australia, marsupials radiated into the wide variety seen today, including not only omnivorous and carnivorous forms such as were present in South America, but also into large herbivores. Modern marsupials appear to have reached the islands of New Guinea and Sulawesi relatively recently via Australia.[67][68][69] A 2010 analysis of retroposon insertion sites in the nuclear DNA of a variety of marsupials has confirmed all living marsupials have South American ancestors. The branching sequence of marsupial orders indicated by the study puts Didelphimorphia in the most basal position, followed by Paucituberculata, then Microbiotheria, and ending with the radiation of Australian marsupials. This indicates that Australidelphia arose in South America, and reached Australia after Microbiotheria split off.[6][7]

In Australia, terrestrial placental mammals disappeared early in the Cenozoic (their most recent known fossils being 55 million-year-old teeth resembling those of condylarths) for reasons that are not clear, allowing marsupials to dominate the Australian ecosystem.[67] Extant native Australian terrestrial placental mammals (such as hopping mice) are relatively recent immigrants, arriving via island hopping from Southeast Asia.[68]

Genetic analysis suggests a divergence date between the marsupials and the placentals at 160 million years ago.[70] The ancestral number of chromosomes has been estimated to be 2n = 14.

A new hypothesis suggests that South American microbiotheres resulted from a back-dispersal from eastern Gondwana due to new cranial and post-cranial marsupial fossils from the Djarthia murgonensis from the early Eocene Tingamarra Local Fauna in Australia that indicate the Djarthia murgonensis is the most plesiomorphic, the oldest unequivocal australidelphian, and may be the ancestral morphotype of the Australian marsupial radiation.[71]
"""


@pytest.mark.parametrize("threshold", np.arange(0, 1, 0.1))
def test_duplications(threshold):
    """
    A generic test covering main usecases.
    """

    hyperparameters = {
        "num_keywords": 10,
        "merge_threshold": threshold,
        "alpha": 0.1,
        "token_prune_len": 3
    }

    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    out_keywords = keyword_detector.find_keywords(example_document,
                                                  input_type="string")

    assert len(out_keywords) == 10
    assert len([x for x in out_keywords if not x]) == 0

    for enx, keyword_one in enumerate(out_keywords):
        if enx + 1 < len(out_keywords):
            k1 = out_keywords[enx][0]
            k2 = out_keywords[enx+1][0]
            if k1 in k2 or k2 in k1:
                raise Exception("Duplicate found")
