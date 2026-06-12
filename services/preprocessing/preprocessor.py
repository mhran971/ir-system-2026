# services/preprocessing/preprocessor.py
import re
import string

class TextPreprocessor:
    """
    معالج النصوص - يقوم بتنظيف وتجهيز النصوص للفهرسة والبحث
    """
    
    def __init__(self, use_stemming=False, use_lemmatization=False):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        # قائمة stop words شائعة في اللغة الإنجليزية
        self.stop_words = {
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
    
    def process(self, text):
        """
        معالجة النص وإرجاع قائمة الكلمات
        """
        if not text or not isinstance(text, str):
            return []
        
        # 1. تحويل إلى حروف صغيرة
        text = text.lower()
        
        # 2. إزالة الأرقام
        text = re.sub(r'[0-9]', ' ', text)
        
        # 3. إزالة علامات الترقيم
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 4. تجزئة النص إلى كلمات
        tokens = text.split()
        
        # 5. إزالة stop words والكلمات القصيرة جداً (أقل من 3 حروف)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # 6. تطبيق stemming بسيط إذا كان مطلوباً
        if self.use_stemming:
            tokens = [self._simple_stem(t) for t in tokens]
        
        return tokens
    
    def _simple_stem(self, word):
        """
        stemming بسيط - إزالة اللواحق الشائعة
        """
        if len(word) < 4:
            return word
        
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'ment', 'tion', 'ness', 'ing']
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        
        return word

# للتوافق مع الكود القديم
SimplePreprocessor = TextPreprocessor