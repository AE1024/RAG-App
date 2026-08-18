"""
DeepEval RAG kalite testleri — Türk Hukuku Chatbot
Çalıştırma: pytest backend/tests/tests_rag.py -v
"""
from dotenv import load_dotenv

load_dotenv()

from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from backend.tests.judge import judge

# ── 1. Faithfulness — LLM sadece bağlamdaki bilgiyi mi kullanıyor? ───────────


def test_faithfulness_passes():
    """Bağlama sadık yanıt yüksek faithfulness skoru almalı."""
    test_case = LLMTestCase(
        input="Nüfus Hizmetleri Kanunu'na göre doğumun bildirilmesi için süre kaç gündür?",
        actual_output=(
            "Nüfus Hizmetleri Kanunu Madde 15'e göre, sağ doğan her çocuğun "
            "doğumdan itibaren 30 gün içinde nüfus müdürlüğüne bildirilmesi zorunludur."
        ),
        retrieval_context=[
            "[Nüfus Hizmetleri Kanunu (5490) — Madde 15]\n"
            "Sağ olarak dünyaya gelen her çocuğun, doğumdan itibaren Türkiye'de "
            "otuz gün içinde nüfus müdürlüğüne bildirilmesi zorunludur."
        ],
    )
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


def test_faithfulness_fails_on_hallucinated_number():
    """Bağlamda '30 gün' yazarken LLM '60 gün' diyorsa faithfulness düşmeli."""
    test_case = LLMTestCase(
        input="Nüfus Hizmetleri Kanunu'na göre doğumun bildirilmesi için süre kaç gündür?",
        actual_output=(
            "Nüfus Hizmetleri Kanunu Madde 15'e göre, sağ doğan her çocuğun "
            "doğumdan itibaren 60 gün içinde nüfus müdürlüğüne bildirilmesi zorunludur."  # YANLIŞ
        ),
        retrieval_context=[
            "[Nüfus Hizmetleri Kanunu (5490) — Madde 15]\n"
            "Sağ olarak dünyaya gelen her çocuğun, doğumdan itibaren Türkiye'de "
            "otuz gün içinde nüfus müdürlüğüne bildirilmesi zorunludur."
        ],
    )
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Halüsinasyon içeren yanıt faithfulness testini geçmemeli. "
        f"Score: {metric.score} | Reason: {metric.reason}"
    )

# ── 2. Answer Relevancy — Yanıt soruyla ilgili mi? ──────────────────────────

def test_answer_relevancy_passes():
    """Soruya doğrudan cevap veren yanıt yüksek answer relevancy skoru almalı."""
    test_case = LLMTestCase(
        input="Tüketicinin Korunması Hakkında Kanun'a göre cayma hakkı süresi kaç gündür?",
        actual_output=(
            "Tüketicinin Korunması Hakkında Kanun Madde 48'e göre, tüketici "
            "on dört gün içinde herhangi bir gerekçe göstermeksizin ve cezai şart "
            "ödemeksizin sözleşmeden cayma hakkına sahiptir."
        ),
        retrieval_context=[
            "[Tüketicinin Korunması Hakkında Kanun (6502) — Madde 48]\n"
            "Tüketici, on dört gün içinde herhangi bir gerekçe göstermeksizin "
            "ve cezai şart ödemeksizin sözleşmeden cayma hakkına sahiptir."
        ],
        
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


def test_answer_relevancy_fails_off_topic():
    """Tamamen konu dışı yanıt düşük answer relevancy skoru almalı."""
    test_case = LLMTestCase(
        input="Tüketicinin Korunması Hakkında Kanun'a göre cayma hakkı süresi kaç gündür?",
        actual_output="Hava bugün çok güzel, dışarıya çıkmak lazım.",  # Alakasız
        retrieval_context=[
            "[Tüketicinin Korunması Hakkında Kanun (6502) — Madde 48]\n"
            "Tüketici, on dört gün içinde cayma hakkına sahiptir."
        ],
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Alakasız yanıt answer relevancy testini geçmemeli. Score: {metric.score}"
    )


# ── 3. Hallucination  ─────────────────


def test_hallucination_low_for_grounded_answer():
    """Bağlama dayalı yanıtta halüsinasyon skoru düşük (iyi) olmalı."""
    test_case = LLMTestCase(
        input="İş Kanunu'na göre işi altı aydan az süren işçi için ihbar süresi nedir?",
        actual_output=(
            "İş Kanunu Madde 17 uyarınca, işi altı aydan az sürmüş işçi için "
            "ihbar süresi iki haftadır."
        ),
        context=[
            "İş sözleşmeleri; işi altı aydan az sürmüş işçi için iki hafta sonra "
            "feshedilmiş sayılır. İşveren bildirim süresine ait ücreti peşin vermek "
            "suretiyle iş sözleşmesini feshedebilir."
        ],
    )
    metric = HallucinationMetric(threshold=0.5, model=judge)
    assert_test(test_case, [metric])


def test_hallucination_high_for_contradicting_content():
    """Context'teki 'iki hafta' yerine 'dört hafta' diyen yanıt hallucination yakalamalı.

    HallucinationMetric ÇELIŞKI (contradiction) arar, yokluğu değil.
    Yanlış konu (kıdem tazminatı) → çelişki yok → score=0.
    Aynı konuda yanlış rakam (iki→dört) → doğrudan çelişki → score yüksek.
    """
    test_case = LLMTestCase(
        input="İş Kanunu'na göre işi altı aydan az süren işçi için ihbar süresi nedir?",
        actual_output=(
            "İş Kanunu Madde 17 uyarınca, işi altı aydan az sürmüş işçi için "
            "ihbar süresi dört haftadır."  # YANLIŞ — context'te "iki hafta" yazıyor
        ),
        context=[
            "İş sözleşmeleri; işi altı aydan az sürmüş işçi için iki hafta sonra "
            "feshedilmiş sayılır."
        ],
    )
    metric = HallucinationMetric(threshold=0.5, model=judge)
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Context'e aykırı yanıt hallucination testini geçmemeli. Score: {metric.score}"
    )


# ── 4. Contextual Precision — Gelen chunk'lar sıralı ve alakalı mı? ──────────


def test_contextual_precision_with_noise():
    """Alakalı chunk önce, alakasız chunk sonra geldiğinde precision yüksek olmalı."""
    test_case = LLMTestCase(
        input="Tüketici cayma hakkını kaç günde kullanabilir?",
        expected_output=(
            "Tüketici, on dört gün içinde herhangi bir gerekçe göstermeksizin "
            "ve cezai şart ödemeksizin sözleşmeden cayma hakkına sahiptir."
        ),
        actual_output=(
            "Tüketicinin Korunması Hakkında Kanun Madde 48'e göre tüketici, "
            "on dört gün içinde cayma hakkını kullanabilir."
        ),
        retrieval_context=[
            # Alakalı chunk
            "[Tüketicinin Korunması Hakkında Kanun (6502) — Madde 48]\n"
            "Tüketici, on dört gün içinde herhangi bir gerekçe göstermeksizin "
            "ve cezai şart ödemeksizin sözleşmeden cayma hakkına sahiptir.",
            # Alakasız chunk (precision'ı test etmek için)
            "[Gelir Vergisi Kanunu (193) — Madde 1]\n"
            "Gerçek kişilerin gelirleri gelir vergisine tabidir.",
        ],
    )
    metric = ContextualPrecisionMetric(threshold=0.5, model=judge)
    assert_test(test_case, [metric])


# ── 5. Contextual Recall — Bağlam, beklenen yanıtı kapsıyor mu? ─────────────


def test_contextual_recall_complete():
    """Bağlam beklenen yanıtın tüm unsurlarını kapsıyorsa recall yüksek olmalı."""
    test_case = LLMTestCase(
        input="Nüfus Hizmetleri Kanunu'na göre doğum bildirimi kim tarafından yapılır?",
        expected_output=(
            "Doğum bildirimi; veli, vasi, kayyım, bunların bulunmaması hâlinde "
            "çocuğun büyük ana veya büyük babası ya da ergin kardeşleri tarafından yapılır."
        ),
        actual_output=(
            "Nüfus Hizmetleri Kanunu'na göre doğum bildirimi veli, vasi veya kayyım "
            "tarafından yapılır; bunların bulunmaması hâlinde çocuğun büyük ana, büyük "
            "baba veya ergin kardeşleri de bildirimde bulunabilir."
        ),
        retrieval_context=[
            "[Nüfus Hizmetleri Kanunu (5490) — Madde 15]\n"
            "Doğum bildirimi; veli, vasi, kayyım, bunların bulunmaması hâlinde "
            "çocuğun büyük ana, büyük baba veya ergin kardeşleri ya da çocuğu "
            "yanında bulunduranlar tarafından yapılır.",
        ],
    )
    metric = ContextualRecallMetric(threshold=0.6, model=judge)
    assert_test(test_case, [metric])


def test_contextual_recall_incomplete_context():
    """Bağlam beklenen yanıtın yalnızca bir bölümünü kapsıyorsa recall düşmeli."""
    test_case = LLMTestCase(
        input=(
            "Türk Borçlar Kanunu'na göre genel zamanaşımı süresi nedir "
            "ve haksız fiillerde bu süre değişiyor mu?"
        ),
        expected_output=(
            "Genel zamanaşımı on yıldır. Haksız fiilden doğan taleplerde ise "
            "zararın ve failin öğrenilmesinden itibaren iki yıl, her hâlde on yıl "
            "geçmekle zamanaşımı dolmaktadır."
        ),
        actual_output="Genel zamanaşımı süresi on yıldır.",
        retrieval_context=[
            # Sadece genel süreyi veriyor; haksız fiil istisnası yok
            "[Türk Borçlar Kanunu (6098) — Madde 146]\n"
            "Kanunda daha kısa bir zamanaşımı süresi öngörülmedikçe, her alacak "
            "on yıllık zamanaşımına tabidir.",
        ],
    )
    metric = ContextualRecallMetric(threshold=0.7, model=judge)
    metric.measure(test_case)
    assert not metric.is_successful(), (
        f"Eksik bağlam yüksek recall skoru almamalı. Score: {metric.score}"
    )


# ── 6. Contextual Relevancy — Gelen chunk'lar soruyla ilgili mi? ─────────────


def test_contextual_relevancy_relevant_chunks():
    """Soruyla doğrudan ilgili chunk yüksek contextual relevancy skoru almalı."""
    test_case = LLMTestCase(
        input="Güvenlik soruşturması kimler hakkında yapılır?",
        actual_output=(
            "Güvenlik Soruşturması ve Arşiv Araştırması Kanunu Madde 3'e göre; "
            "statüsü veya çalıştırma şekline bağlı olmaksızın ilk defa veya yeniden "
            "kamu görevine atanacaklar hakkında yapılır."
        ),
        retrieval_context=[
            "[Güvenlik Soruşturması ve Arşiv Araştırması Kanunu (7315) — Madde 3]\n"
            "Statüsü veya çalıştırma şekline bağlı olmaksızın ilk defa veya yeniden "
            "memuriyete yahut kamu görevine atanacaklar hakkında güvenlik soruşturması "
            "ve arşiv araştırması yapılır.",
        ],
    )
    metric = ContextualRelevancyMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


# ── 7. GEval — Domain-specific değerlendirmeler ──────────────────────────────


def test_geval_legal_citation():
    """Yanıtta kanun tam adı ve madde numarası birlikte yer almalı."""
    test_case = LLMTestCase(
        input="İş Kanunu'nda deneme süresi kaç aydır?",
        actual_output=(
            "İş Kanunu Madde 15 uyarınca, taraflarca iş sözleşmesine en fazla iki aylık "
            "deneme süresi konulabilir. Deneme süresi içinde taraflar iş sözleşmesini "
            "bildirimsiz ve tazminatsız feshedebilir."
        ),
        retrieval_context=[
            "[İş Kanunu (4857) — Madde 15]\n"
            "Taraflarca iş sözleşmesine iki aylık deneme süresi konulabilir. "
            "Deneme süresi içinde taraflar iş sözleşmesini bildirimsiz ve "
            "tazminatsız feshedebilir."
        ],
    )
    metric = GEval(
        name="LegalCitationCompleteness",
        criteria=(
            "Yanıtta ilgili Türk kanununun tam adı (örn: 'İş Kanunu') VE madde numarası "
            "(örn: 'Madde 15') açıkça belirtilmiş mi? "
            "Her ikisi varsa yüksek puan, biri eksikse orta puan, ikisi de yoksa düşük puan ver."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=judge,
    )
    assert_test(test_case, [metric])


def test_geval_disclaimer_present():
    """Yanıt 'hukuki tavsiye niteliği taşımaz' uyarısı içermeli."""
    test_case = LLMTestCase(
        input="İşten çıkarıldım, ne yapmalıyım?",
        actual_output=(
            "İş Kanunu Madde 17 gereğince işverenin yasal ihbar sürelerine uyması gerekir. "
            "Ancak bu bilgiler yalnızca kanun metinlerini aktarmakta olup hukuki tavsiye "
            "niteliği taşımamaktadır. Somut durumunuz için bir hukuk uzmanına başvurmanızı öneririm."
        ),
        retrieval_context=[
            "[İş Kanunu (4857) — Madde 17]\n"
            "İş sözleşmeleri; işi altı aydan az sürmüş işçi için iki hafta sonra "
            "feshedilmiş sayılır."
        ],
    )
    metric = GEval(
        name="DisclaimerPresence",
        criteria=(
            "Yanıt, sistemin yalnızca kanun metinlerini aktardığını ve hukuki tavsiye "
            "vermediğini belirten bir uyarı veya feragatname içeriyor mu? "
            "İçeriyorsa yüksek, içermiyorsa düşük puan ver."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.6,
        model=judge,
    )
    assert_test(test_case, [metric])


def test_geval_no_negative_inversion():
    """'Sayılmaz', 'yasaktır' gibi olumsuzluklar tersine çevrilmemeli."""
    test_case = LLMTestCase(
        input="Sosyal Sigortalar Kanunu'na göre hangi çalışanlar sigortalı sayılmaz?",
        actual_output=(
            "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu'na göre, "
            "işverenin ücretsiz çalışan eşi sigortalı sayılmaz."
        ),
        retrieval_context=[
            "[Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu (5510) — Madde 6]\n"
            "İşverenin ücretsiz çalışan eşi bu Kanunun uygulanmasında sigortalı sayılmaz."
        ],
    )
    metric = GEval(
        name="NegationPreservation",
        criteria=(
            "Yanıt, bağlamdaki 'sayılmaz', 'hariç', 'yasaktır' gibi olumsuzluk ifadelerini "
            "doğru koruyor mu, yoksa anlamı tersine mi çeviriyor? "
            "Olumsuzluk korunmuşsa yüksek puan, tersine çevrilmişse düşük puan ver."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        threshold=0.7,
        model=judge,
    )
    assert_test(test_case, [metric])
