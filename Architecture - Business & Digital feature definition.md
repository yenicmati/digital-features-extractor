# 

# Digital feature concept

## A quel problème la notion de feature doit répondre chez ADEO en 2026?

Parler de  la valeur qu’on va créer plutôt que de parler des fonctionnalités techniques que l’on va déployer  
Ex : augmenter le taux de conversion quand on déploie le subset publication. 

Produire des changelog explicites pour les BU

Éclairer le contenu d’une initiative et sa valeur au travers des features. 

Être plus en maîtrise de ce qu’on livre, et de la couverture des tests à réaliser

## Comment est utilisée la notion de feature dans les produits digitaux aujourd’hui ?

C’est une terminologie utilisée mais pas vraiment implémentée dans les outils  
La BU a demandé telle feature ? Quand est-ce que cette feature arrive ?

## 

## Proposition de Définition : 

Une business feature est une  capacité ou fonctionnalité spécifique d'un ou plusieurs **Digital Product** qui apporte de la valeur à l'utilisateur final en répondant à un besoin précis. Une **Digital Feature** est un élément de la roadmap d'un **Digital Product** et peut être développée, déployée et mesurée de manière indépendante.

La **Digital Feature** est le résultat de l’implémentation technique qui répond à un besoin métier. 

La **Digital Feature** répond à un problème utilisateur et constitue donc une réponse à un besoin métier qui apporte une valeur à l’entreprise. Elle **permet de distinguer le Digital product** sur le marché ou au sein des autres digital products.

Alors que le **Digital Product** définit l'offre de valeur globale, la **Digital Feature** permet un pilotage granulaire de la valeur.

La **Digital Feature** est aussi un élément du changelog.

Une **Feature** n'est pas "juste une ligne de code", mais elle nécessite des composants techniques (API, Front, Back) pour fonctionner. Elle est distincte de la fonction technique. 

*Prendre des exemples front ET back. Exemple c’est quoi les features d’un produit comme offer repository.*  
*Attention à la granularité (indépendance de la feature ?)*  
*Notion de Persona ?*

## Proposition d’Attributs

| Attribut  | Définition  |
| :---- | :---- |
| **Digital Feature's Identifier** | Identifiant unique de la fonctionnalité au sein du Digital Product. |
| **Digital Feature's Name** | Nom officiel de la fonctionnalité, qui va servir à décrire la couverture fonctionnelle du produit. Elle est comprise par les utilisateurs finaux et par l’équipe produit. |
| **Digital Feature's Description** | Description du but de la fonctionnalité et de la valeur qu'elle apporte. |
| **Digital Feature's Status** | Statut de la fonctionnalité dans son cycle de vie (ex: To Be Developed, Work in progress, Live, Deprecated). La fonctionnalité peut être activable ou non dans un contexte BU. Le statut de la fonctionnalité peut être différent d’une version à l’autre du produit. |
| **Parent Digital Product Identifier** | Identifiant du Digital Product auquel cette fonctionnalité est rattachée. |
| **Associated Business Capability Identifier** | Référence à la *Business Capability* parente. |
| **activable or not** | YES/NO/MANDATORY |
| **Activation context** | en autonomie, niveau région,... |

## ANNEXE 1 : LA NOTION DE FEATURE SUR LE MARCHE

la notion de **feature** (fonctionnalité) a évolué pour devenir une unité de valeur stratégique plutôt qu'une simple ligne de code. Elle ne se définit plus seulement par ce qu'elle "fait" techniquement, mais par le problème qu'elle résout pour l'utilisateur.

Voici comment elle est définie et utilisée aujourd'hui :

### **1\. Définition orientée "Valeur" et "Impact"**

Aujourd'hui, une feature est considérée comme une capacité distincte d'un produit qui apporte une valeur commerciale ou résout un besoin client précis.

* **Différence entre Feature et Fonction :** La **feature** est ce que l'utilisateur manipule directement (ex: "Paiement mobile"), tandis que la **fonction** est la composante technique invisible qui la supporte (ex: "Appel API vers la banque").  
* **Lien avec la stratégie :** Elle sert de brique élémentaire pour construire la proposition de valeur d'un produit et le différencier sur le marché.

### **2\. Utilisation dans le Pilotage Produit**

La feature est l'unité de mesure centrale dans les roadmaps modernes.

* **Priorisation par la valeur :** Les Product Managers ne listent plus des fonctionnalités par intuition, mais les priorisent en fonction de leur impact potentiel sur le business et de leur faisabilité.  
* **Évitement de la "Feature Factory" :** Le marché s'éloigne du modèle où l'on produit des fonctionnalités "à la chaîne" sans mesurer leur utilité réelle. L'accent est mis sur le cycle "conception $\\to$ livraison $\\to$ usage" pour garantir que chaque feature améliore réellement la chaîne de valeur.

### **3\. Pratiques de Déploiement : Feature Management**

L'utilisation technique a également changé avec l'essor du **Feature Management** :

* **Feature Flagging :** On déploie une fonctionnalité sans l'activer pour tout le monde. Cela permet de tester son adoption sur un petit groupe avant un lancement global.  
* **Feature Experimentation :** Les équipes lancent différentes versions d'une même feature pour identifier, grâce aux données, celle qui génère les meilleurs résultats (A/B testing granulaire).

Feedbacks Loïc:

| Source / Contexte | Formulation de la définition (synthèse) | Points clés caractéristiques |
| :---- | :---- | :---- |
| Airfocus (glossaire produit) | Partie d’un logiciel qui fournit une capacité ou fonction précise contribuant au but global du logiciel.[airfocus](https://airfocus.com/glossary/what-is-software-feature/) | Fonctionnalité précise, contribue au but du produit, unité de planification produit. |
| LaunchNotes (produit / ops) | Fonctionnalités et composants spécifiques qui composent une application et permettent d’accomplir des tâches.[launchnotes](https://www.launchnotes.com/glossary/software-feature-in-product-management-and-operations) | Fonctionnalités concrètes, liées à des tâches utilisateur, structurent l’expérience globale. |
| Usersnap (Agile) | Fonctionnalité ou capacité utilisateur distincte, orientée valeur, souvent côté client, livrable et testable.[usersnap](https://usersnap.com/glossary/agile-feature) | Distincte, centrée valeur client, livrable de façon indépendante, souvent représentée dans le backlog. |
| GeeksforGeeks (Feature vs User Story) | Caractéristique ou fonctionnalité précieuse d’un produit qui satisfait un besoin client et sert les objectifs business.[geeksforgeeks](https://www.geeksforgeeks.org/software-engineering/what-is-the-difference-between-a-feature-and-a-user-story/) | Caractère distinctif, valeur client \+ business, élément qui différencie des versions/produits. |
| Wikipédia / littérature académique | Aspect, qualité ou caractéristique distinctive et visible par l’utilisateur d’un système logiciel.[wikipedia](https://en.wikipedia.org/wiki/Feature-rich) | Aspect “user-visible”, caractéristique distinctive, peut être fonctionnelle ou non-fonctionnelle (perf, portabilité, etc.). |
| IEEE 829 (test logiciel, via Wikipédia) | Caractéristique distinctive d’un élément logiciel (performance, portabilité, fonctionnalité, etc.).[wikipedia](https://en.wikipedia.org/wiki/Feature-rich) | Inclut aussi des dimensions non fonctionnelles, point de référence pour la conception de tests. |
| 6B Systems (développement logiciel) | Capacité ou fonction permettant aux utilisateurs de réaliser des actions pour obtenir un résultat souhaité.[6b](https://6b.systems/insight/what-is-a-feature-in-software-development/) | Orienté action utilisateur et résultat, peut être simple (recherche) ou complexe (système de planning). |
| Tres Astronautas (vulgarisation non-tech) | Élément de fonctionnalité/capacité distincte qui apporte de la valeur, comme “recherche” ou “checkout”.[tresastronautas](https://www.tresastronautas.com/en/dictionary/feature) | Métaphore “application” dans l’application, valorise la compréhension par les non-tech, axé bénéfice utilisateur. |
| GeeksforGeeks (product feature) | Attribut ou fonction d’un produit qui apporte de la valeur au client et le différencie sur le marché.[geeksforgeeks](https://www.geeksforgeeks.org/software-engineering/what-are-product-features-definition-features-and-examples/) | Notion marketing/produit, valeur client, différenciation, peut combiner composants fonctionnels ou améliorations de performance. |
| StarAgile / blogs Agile (hiérarchie) | Fonctionnalité distincte apportant de la valeur, de portée modérée, livrable en quelques sprints.[staragile+1](https://staragile.com/blog/epic-vs-feature-vs-user-story) | Niveau intermédiaire (entre epic et user stories), orientée valeur, testable et potentiellement publiable telle quelle. |
| Scrum.org / outils Agile (Azure DevOps) | Résultat fonctionnel vers lequel convergent plusieurs user stories, dans une optique de solution produit.[scrum+1](https://www.scrum.org/forum/scrum-forum/6221/features-vs-user-stories) | Sert à regrouper et donner du sens à des stories, exprimé en termes de problème/solution plutôt que tâches techniques. |
| Waterloo GSD Lab (product lines) | Caractéristiques fonctionnelles et non fonctionnelles qui distinguent les produits d’une même famille.[gsd.uwaterloo](https://gsd.uwaterloo.ca/featureStudy.html) | Unité de variabilité/réutilisation, utilisée pour composer/distinguer des produits dans une ligne de produits. |

* Fonctionnalité ou capacité distincte : une feature est vue comme une unité cohérente de comportement ou de caractéristique, et pas un simple “bout de tâche”.   
* Valeur et résultat utilisateur : quasiment toutes les définitions associent la feature à un besoin, une tâche ou un bénéfice utilisateur explicite.   
* Rôle produit / organisationnel : en Agile, la feature sert de niveau de découpage intermédiaire, regroupant plusieurs user stories autour d’un même objectif fonctionnel.   
* Portée fonctionnelle et parfois non-fonctionnelle : certaines définitions incluent la performance, la sécurité, la portabilité comme “features” au sens qualité ou attribut distinctif. 

