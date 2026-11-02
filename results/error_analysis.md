# F6 — Lecture assistée de 30 erreurs (provisoire)

Étude verrouillée le 29/10/2026 à 15:58 UTC ; candidat `A_char_s42`, données officielles figées. Les 30 identifiants correspondent à `f6_30_erreurs.csv`. **Cette lecture est une proposition d'analyse, pas une validation humaine des labels.** Les catégories peuvent se cumuler ; un désaccord entre annotateurs ne démontre pas que le label officiel est faux.

Le candidat utilise des n-grammes de caractères : l'étiquette automatique « tokenisation Arabizi possible » doit donc être comprise comme une **hypothèse de couverture des graphies**, et non comme la preuve d'un échec WordPiece. Les catégories automatiques « ironie ou autre » ne prouvent aucune ironie. La lecture ci-dessous précise ces limites et retrouve des indices de français dans des exemples initialement rangés parmi les confusions neutre/polarisé.

| Ligne test | Label → prédiction | Catégories proposées | Interprétation possible à valider |
|---|---|---|---|
| 1 | negative → neutral | Arabizi ; désaccord d'annotation ; neutre/polarisé | Le contraste entre une apparence bienveillante et un coup préparé porte un jugement négatif. Le rire initial ne suffit pas à déterminer la polarité. |
| 6 | negative → neutral | Arabizi ; neutre/polarisé | Interpellation et vocabulaire péjoratif en Arabizi. Vérifier la couverture de ces graphies avant d'attribuer l'erreur à un mécanisme précis. |
| 3 | positive → negative | Cible ; ironie non établie | « دولة الحق و القانون » est littéralement élogieux. Une lecture ironique demanderait le contexte ; le mot France peut aussi être associé à des opinions variées. |
| 330 | positive → negative | Français/arabe ; cible ; annotation à discuter | Alternance avec « des profil ». Le texte alerte contre des provocations ; l'interprétation positive officielle demande de préciser la cible de l'opinion. |
| 9 | negative → neutral | Arabizi ; ellipse ; désaccord d'annotation | Arabizi familier et référence elliptique à un interlocuteur. Le contexte conversationnel manque pour une interprétation sûre. |
| 16 | negative → neutral | Arabizi ; neutre/polarisé | « 3ar » suggère un reproche. La graphie et le nom propre peuvent rendre cette évaluation difficile à reconnaître. |
| 5 | negative → positive | Sentiments concurrents | La critique « دون مستوى » coexiste avec le souhait « نتمنى ». Un indice de souhait positif peut concurrencer l'évaluation négative de l'adversaire. |
| 510 | positive → negative | Question rhétorique ; cible ; annotation à discuter | Le texte évoque un événement grave et interpelle une personne. La polarité positive officielle ne se déduit pas simplement de mots positifs isolés. |
| 17 | negative → neutral | Ironie possible ; Arabizi ; désaccord d'annotation | Détournement possible du proverbe du lever tôt : de l'or est promis mais « maban lia walo » indique que rien n'est apparu. L'ironie reste à confirmer. |
| 24 | negative → neutral | Français ; négation/contexte ; humour possible | « salut » et « kanbghik » apparaissent dans un contexte moqueur et de reproche. Les mots affectifs ou le rire peuvent être trompeurs isolément. |
| 7 | positive → negative | Cible politique ; lexique concurrent | Le texte peut approuver une stratégie tout en critiquant un adversaire. Des termes péjoratifs dirigés vers celui-ci ne déterminent pas seuls la polarité globale. |
| 657 | negative → positive | Citation ; ambiguïté ; Arabizi | Citation attribuée par « sic ». Le ton et le contexte manquent ; la polarité négative officielle ne doit pas être réinterprétée automatiquement. |
| 23 | negative → neutral | Français ; moquerie ; désaccord d'annotation | « nostalgique », « après le mur de Berlin » et les rires construisent une moquerie possible. Les emojis rieurs ne signifient pas nécessairement un sentiment positif. |
| 26 | negative → neutral | Négation Arabizi ; neutre/polarisé | « mafikomch ti9a » exprime un manque de confiance. Le modèle peut mal couvrir ses variantes graphiques ; cela reste une hypothèse à tester. |
| 77 | positive → negative | Formule politique ; contexte ; annotation à discuter | Formule protestataire et hashtag fortement contextuels. La lecture isolée ne suffit pas pour remplacer le label positif officiel. |
| 760 | positive → negative | Mobilisation ; artefact de corpus | Appel aux militants « الأحرار ». Le marqueur littéral `[[positiveHOTO]]` est présent dans le fichier brut officiel ; il doit être signalé, sans correction rétroactive du benchmark. |
| 25 | positive → neutral | Français ; cible ; désaccord d'annotation | L'éloge « tu es magnifique » est suivi de mises en garde et de critiques de tiers. Le sentiment dépend de la personne visée. |
| 27 | negative → neutral | Arabizi ; neutre/polarisé | « 3ssbuni », « mochkil » et l'idée d'affection feinte fournissent des indices négatifs. La phrase reste conversationnelle et elliptique. |
| 92 | positive → negative | Pragmatique ; contexte politique | Fin d'un appel mobilisateur et formule religieuse. L'interprétation positive peut être pragmatique plutôt que lexicale. |
| 848 | positive → negative | Cibles opposées ; artefact de corpus | Critique de l'État mais soutien possible aux manifestations. Le fragment littéral `COpositive22` constitue aussi un artefact visible du corpus. |
| 50 | negative → neutral | Conversation de fans ; requête ; désaccord d'annotation | Demande insistante de rester à l'écart d'un conflit. La politesse de la requête ne suffit pas à rendre le texte positif. |
| 30 | negative → neutral | Français ; sentiments concurrents | Présence de « mais », encouragement initial puis réprimande. Une décision globale doit tenir compte de l'ensemble de l'énoncé. |
| 93 | positive → negative | Événement vs opinion ; annotation à discuter | Annonce d'une exclusion politique apparemment factuelle. Le label positif pourrait refléter une approbation contextuelle ; cette hypothèse exige une validation. |
| 893 | positive → negative | Éloge ; question rhétorique ; artefact | « قمة التحضر » et « بنظام و انتظام » peuvent porter l'éloge malgré la critique des contradicteurs. Le marqueur `[[positiveHOTO]]` est également présent. |
| 55 | positive → neutral | Conversation ; encouragement ; désaccord d'annotation | Rire et formules de patience. La positivité officielle peut reposer sur un encouragement conversationnel, sans être évidente hors contexte. |
| 37 | negative → neutral | Reproche ; Arabizi | « 5ram » et « dnoub » suggèrent un reproche moral. Les chiffres font partie de la graphie et ne doivent pas être supprimés. |
| 97 | positive → negative | Éloge ; ironie non établie | « دولة الحق و القانون » et « انتا معلم » sont des indices élogieux. Une éventuelle ironie reste indémontrée sans contexte. |
| 1246 | positive → negative | Français ; soutien ; cibles multiples | Soutien de fans et « grand bisou » coexistent avec des critiques et des conditions. Les négations ne s'appliquent pas toutes à la même cible. |
| 60 | positive → neutral | Référents absents ; désaccord d'annotation | Conversation de fans avec référents absents. Le rire seul ne suffit pas à justifier un label ; lecture humaine nécessaire. |
| 38 | positive → neutral | Formule sociale ; neutre/polarisé | « labass 3lik » et des hashtags peuvent fonctionner comme une formule de soutien. La frontière entre simple prise de nouvelles et positivité est discutable. |

**Conclusion provisoire :** plusieurs erreurs impliquent la cible du sentiment, le contexte conversationnel, des graphies Arabizi variables et du français. Les données ne permettent pas de transformer ces hypothèses en causes démontrées. Conserver les labels officiels et demander une validation humaine ; ne pas réentraîner à partir de ces erreurs test dans cette étude.
