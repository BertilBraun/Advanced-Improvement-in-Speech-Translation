import random
from src.logger_utils import get_logger
from src.datasets.util import iterate_over_dataset

from src.datasets.base.mt_dataset import MTDataset, TextSample

logger = get_logger('Paraphrases::Punctuation Enhancement Dataset')


class PunctuationEnhancementDataset(MTDataset):
    def __init__(self, datasource: MTDataset) -> None:
        self.datasource = datasource
        # Expanded mapping of words to their potential ASR misinterpretations
        self.asr_noise_map = {
            'accept': ['except', 'expect'],
            'bare': ['bear'],
            'brake': ['break'],
            'cell': ['sell'],
            'flower': ['flour'],
            'hear': ['here'],
            'hole': ['whole'],
            'knight': ['night'],
            'mail': ['male'],
            'one': ['won'],
            'pair': ['pear', 'pare'],
            'peace': ['piece'],
            'plain': ['plane'],
            'right': ['write', 'rite'],
            'sea': ['see'],
            'sun': ['son'],
            'their': ['there', 'they’re'],
            'to': ['too', 'two'],
            'wear': ['where', 'ware'],
            'week': ['weak'],
            'affect': ['effect'],
            'altar': ['alter'],
            'billed': ['build'],
            'board': ['bored'],
            'by': ['buy', 'bye'],
            'complement': ['compliment'],
            'council': ['counsel'],
            'die': ['dye'],
            'fair': ['fare'],
            'grate': ['great'],
            'principal': ['principle'],
            'rain': ['reign', 'rein'],
            'role': ['roll'],
            'stair': ['stare'],
            'stationary': ['stationery'],
            'tail': ['tale'],
            'threw': ['through'],
            'vain': ['vane', 'vein'],
            'waste': ['waist'],
            'aid': ['aide'],
            'air': ['heir', 'err'],
            'altar': ['alter'],
            'arc': ['ark'],
            'ball': ['bawl'],
            'band': ['banned'],
            'beach': ['beech'],
            'beau': ['bow'],
            'beer': ['bear'],
            'berry': ['bury'],
            'billed': ['build'],
            'blue': ['blew'],
            'board': ['bored'],
            'bold': ['bowled'],
            'bough': ['bow'],
            'boy': ['buoy'],
            'bread': ['bred'],
            'break': ['brake'],
            'buy': ['by', 'bye'],
            'capital': ['capitol'],
            'cast': ['caste'],
            'cent': ['scent', 'sent'],
            'cereal': ['serial'],
            'chord': ['cord'],
            'cite': ['sight', 'site'],
            'clause': ['claws'],
            'coal': ['cole'],
            'complement': ['compliment'],
            'coarse': ['course'],
            'core': ['corps'],
            'cue': ['queue'],
            'dear': ['deer'],
            'die': ['dye'],
            'draft': ['draught'],
            'dual': ['duel'],
            'fare': ['fair'],
            'feat': ['feet'],
            'flour': ['flower'],
            'for': ['four', 'fore'],
            'forth': ['fourth'],
            'foul': ['fowl'],
            'genes': ['jeans'],
            'gilt': ['guilt'],
            'grate': ['great'],
            'groan': ['grown'],
            'guessed': ['guest'],
            'hail': ['hale'],
            'hair': ['hare'],
            'hall': ['haul'],
            'heal': ['heel', 'he’ll'],
            'hear': ['here'],
            'hi': ['high'],
            'hole': ['whole'],
            'hour': ['our'],
            'idle': ['idol'],
            'in': ['inn'],
            'jam': ['jamb'],
            'knew': ['new', 'gnu'],
            'knot': ['not', 'naught'],
            'knows': ['nose'],
            'lays': ['laze'],
            'lead': ['led'],
            'leased': ['least'],
            'lessen': ['lesson'],
            'lie': ['lye'],
            'loan': ['lone'],
            'made': ['maid'],
            'mail': ['male'],
            'main': ['mane'],
            'maize': ['maze'],
            'meat': ['meet', 'mete'],
            'miner': ['minor'],
            'moan': ['mown'],
            'mood': ['mooed'],
            'moose': ['mousse'],
            'morning': ['mourning'],
            'muscle': ['mussel'],
            'none': ['nun'],
            'oar': ['or', 'ore'],
            'ode': ['owed'],
            'overseas': ['oversees'],
            'paced': ['paste'],
            'pain': ['pane'],
            'pair': ['pear', 'pare'],
            'pale': ['pail'],
            'passed': ['past'],
            'patience': ['patients'],
            'pause': ['paws'],
            'peace': ['piece'],
            'peak': ['peek', 'peke'],
            'peer': ['pier'],
            'plane': ['plain'],
            'plate': ['plait'],
            'pole': ['poll'],
            'poor': ['pore', 'pour'],
            'pray': ['prey'],
            'principal': ['principle'],
            'profit': ['prophet'],
            'rack': ['wrack'],
            'rain': ['reign', 'rein'],
            'raise': ['rays', 'raze'],
            'read': ['reed', 'red'],
            'real': ['reel'],
            'right': ['write', 'rite'],
            'ring': ['wring'],
            'road': ['rode', 'rowed'],
            'role': ['roll'],
            'root': ['route'],
            'rose': ['rows'],
            'sale': ['sail'],
            'scene': ['seen'],
            'seam': ['seem'],
            'seas': ['sees', 'seize'],
            # ... and many more
        }

        super().__init__(datasource.split)

    def _generate_noisy_samples(self, sentence: str, samples_to_generate: int = 10) -> list[str]:
        noisy_samples = []
        words = sentence.split()

        for _ in range(samples_to_generate):
            new_sentence = words.copy()

            for i, word in enumerate(words):
                if word.lower() in self.asr_noise_map and random.random() < 0.3:  # 30% chance to replace a word
                    new_sentence[i] = random.choice(self.asr_noise_map[word.lower()])

            noisy_samples.append(' '.join(new_sentence))

        noisy_samples = [noisy_sample for noisy_sample in noisy_samples if noisy_sample != sentence]
        logger.debug(f'Generated {len(noisy_samples)} noisy samples for sentence: {sentence}')
        return noisy_samples

    def _load_data(self) -> list[TextSample]:
        data = []

        for cleaned, with_punctuation in iterate_over_dataset(self.datasource, desc='Loading punctuation dataset'):
            data.append((cleaned, with_punctuation))

            for noisy_version in self._generate_noisy_samples(cleaned):
                data.append((noisy_version, with_punctuation))

        return data
