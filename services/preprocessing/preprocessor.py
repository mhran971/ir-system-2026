import re
import string

class TextPreprocessor:

    
    def __init__(self, use_stemming=False, use_lemmatization=False):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        # قائمة stop words شائعة
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
            'this', 'that', 'these', 'those', 'am', 'do', 'does', 'did',
            'doing', 'have', 'having', 'the', 'and', 'or', 'but', 'not',
            'so', 'such', 'can', 'could', 'would', 'should', 'may', 'might',
            'must', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'only', 'own', 'same', 'than', 'then', 'too', 'very',
            'just', 'but', 'do', 'does', 'did', 'doing', 'is', 'are', 'was'
        }
    
    def process(self, text):
        """
        معالجة النص وإرجاع قائمة الكلمات
        """
        if not text or not isinstance(text, str):
            return []
        
        # 1. تحويل إلى حروف صغيرة
        text = text.lower()
        
        # 2. إزالة علامات الترقيم والأرقام
        text = re.sub(r'[0-9]', ' ', text)  # إزالة الأرقام
        text = text.translate(str.maketrans('', '', string.punctuation))  # إزالة علامات الترقيم
        
        # 3. تجزئة النص إلى كلمات
        tokens = text.split()
        
        # 4. إزالة stop words والكلمات القصيرة جداً
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # 5. تطبيق stemming بسيط إذا كان مطلوباً
        if self.use_stemming:
            tokens = [self._simple_stem(t) for t in tokens]
        
        return tokens
    
    def _simple_stem(self, word):
        """
        Stemming بسيط (إزالة اللواحق الشائعة)
        """
        if len(word) < 4:
            return word
        
        # إزالة اللواحق الشائعة
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'ment', 'tion', 'ness']
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        
        return word

# للحفاظ على التوافق مع الكود القديم
SimplePreprocessor = TextPreprocessor