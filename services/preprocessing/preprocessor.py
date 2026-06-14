# services/preprocessing/preprocessor.py
import re
import string
import logging
from typing import List, Set, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import nltk

# تحميل البيانات المطلوبة (مرة واحدة فقط)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger')
    except Exception:
        nltk.download('averaged_perceptron_tagger_eng')

# إعداد logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tags to WordNet POS tags."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class TextPreprocessor:
    """
    معالج النصوص الاحترافي
    
    يقوم بـ:
    - تنظيف النصوص
    - إزالة المحارف الخاصة
    - تقسيم النصوص لكلمات
    - إزالة كلمات التوقف
    - Stemming أو Lemmatization
    - معالجة الأخطاء
    
    مثال الاستخدام:
    >>> preprocessor = TextPreprocessor(use_lemmatization=True)
    >>> tokens = preprocessor.process("The quick brown fox jumps!")
    >>> print(tokens)
    ['quick', 'brown', 'fox', 'jump']
    """
    
    def __init__(
        self,
        language: str = 'english',
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        min_token_length: int = 3,
        remove_numbers: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        تهيئة معالج النصوص
        
        Args:
            language: اللغة (english, arabic, etc.)
            use_stemming: استخدام Stemming
            use_lemmatization: استخدام Lemmatization
            min_token_length: الحد الأدنى لطول الكلمة
            remove_numbers: إزالة الأرقام
            lowercase: تحويل لحروف صغيرة
            remove_punctuation: إزالة علامات الترقيم
            remove_stopwords: إزالة كلمات التوقف
            custom_stopwords: إضافة كلمات توقف مخصصة
        """
        self.language = language
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        
        # تحميل كلمات التوقف
        try:
            self.stop_words = set(stopwords.words(language))
            logger.info(f"✅ تم تحميل {len(self.stop_words)} كلمة توقف للغة {language}")
        except Exception as e:
            logger.warning(f"⚠️ لم يتمكن من تحميل stop words: {e}")
            self.stop_words = self._get_default_stopwords()
        
        # إضافة كلمات توقف مخصصة
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # تهيئة Stemmer و Lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # إحصائيات
        self.stats = {
            'texts_processed': 0,
            'total_tokens': 0,
            'removed_stopwords': 0,
            'removed_short_tokens': 0
        }
    
    def process(self, text: str) -> List[str]:
        """
        معالجة النص وإرجاع قائمة الكلمات
        
        Args:
            text: النص المراد معالجته
            
        Returns:
            قائمة الكلمات المعالجة
            
        Raises:
            ValueError: إذا كان المدخل غير صحيح
        """
        # التحقق من صحة المدخل
        if not isinstance(text, str):
            logger.warning(f"⚠️ المدخل ليس نصاً: {type(text)}")
            raise ValueError(f"Expected str, got {type(text)}")
        
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            # 1. تحويل لحروف صغيرة
            if self.lowercase:
                text = text.lower()
            
            # 2. إزالة الأرقام
            if self.remove_numbers:
                text = re.sub(r'\d+', ' ', text)
            
            # 3. إزالة علامات الترقيم
            if self.remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            # 4. إزالة المسافات الزائدة
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 5. تقسيم النص لكلمات
            tokens = word_tokenize(text)
            
            # POS tagging on the full list of tokens to preserve sentence context
            if self.use_lemmatization and not self.use_stemming:
                tagged_tokens = pos_tag(tokens)
            else:
                tagged_tokens = [(t, '') for t in tokens]
            
            # 6. تصفية الكلمات
            filtered_tagged_tokens = []
            for token, tag in tagged_tokens:
                # التحقق من الطول
                if len(token) < self.min_token_length:
                    self.stats['removed_short_tokens'] += 1
                    continue
                
                # إزالة كلمات التوقف
                if self.remove_stopwords and token in self.stop_words:
                    self.stats['removed_stopwords'] += 1
                    continue
                
                filtered_tagged_tokens.append((token, tag))
            
            # 7. تطبيق Stemming أو Lemmatization
            if self.use_stemming:
                processed_tokens = [self._stem(t[0]) for t in filtered_tagged_tokens]
            elif self.use_lemmatization:
                processed_tokens = [
                    self._lemmatize(word, get_wordnet_pos(tag))
                    for word, tag in filtered_tagged_tokens
                ]
            else:
                processed_tokens = [t[0] for t in filtered_tagged_tokens]
            
            # تحديث الإحصائيات
            self.stats['texts_processed'] += 1
            self.stats['total_tokens'] += len(processed_tokens)
            
            return processed_tokens
        
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة النص: {e}")
            raise
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        معالجة مجموعة من النصوص
        
        Args:
            texts: قائمة النصوص
            
        Returns:
            قائمة قوائم الكلمات المعالجة
        """
        logger.info(f"📊 بدء معالجة {len(texts)} نصاً")
        results = []
        
        for i, text in enumerate(texts):
            try:
                tokens = self.process(text)
                results.append(tokens)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"   ✅ تمت معالجة {i + 1}/{len(texts)} نصاً")
            except Exception as e:
                logger.warning(f"   ⚠️ خطأ في النص #{i}: {e}")
                results.append([])
        
        logger.info(f"✅ اكتملت معالجة {len(texts)} نصاً")
        return results
    
    def _stem(self, word: str) -> str:
        """تطبيق Stemming على الكلمة"""
        try:
            return self.stemmer.stem(word)
        except Exception as e:
            logger.warning(f"⚠️ خطأ في Stemming: {e}")
            return word
    
    def _lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        """تطبيق Lemmatization على الكلمة"""
        try:
            if pos:
                return self.lemmatizer.lemmatize(word, pos)
            return self.lemmatizer.lemmatize(word)
        except Exception as e:
            logger.warning(f"⚠️ خطأ في Lemmatization: {e}")
            return word
    
    def _get_default_stopwords(self) -> Set[str]:
        """الحصول على قائمة افتراضية من كلمات التوقف"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
            'this', 'that', 'these', 'those', 'am', 'do', 'does', 'did',
            'doing', 'have', 'having', 'or', 'but', 'not', 'so', 'such',
            'can', 'could', 'would', 'should', 'may', 'might', 'must',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'only', 'own', 'same', 'than', 'then', 'too', 'very', 'just'
        }
    
    def get_stats(self) -> dict:
        """الحصول على إحصائيات المعالجة"""
        return {
            'texts_processed': self.stats['texts_processed'],
            'total_tokens': self.stats['total_tokens'],
            'removed_stopwords': self.stats['removed_stopwords'],
            'removed_short_tokens': self.stats['removed_short_tokens'],
            'avg_tokens_per_text': (
                self.stats['total_tokens'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0
                else 0
            )
        }
    
    def print_stats(self) -> None:
        """طباعة الإحصائيات"""
        stats = self.get_stats()
        print("\n📊 إحصائيات المعالجة:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50)


# للتوافق مع الكود القديم
SimplePreprocessor = TextPreprocessor