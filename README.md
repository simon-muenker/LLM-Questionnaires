# LLM Questionnaires: Neural Anthropology for Gen. AI

This repository contains the code and resources for our comprehensive study on intrinsic bias in Large Language Models (LLMs) and their capacity to represent diverse ideological and cultural perspectives through in-context prompting. We systematically evaluate the consistency with which LLMs respond to standardized psychological questionnaires under various minimal persona prompts, benchmarking their responses against robust human survey data. Our experimental design encompasses multiple state-of-the-art open-source models, with each model-persona pairing subjected to k repeated trials functioning as synthetic surveys. Our findings reveal significant variations in response consistency across different model architectures and highlight fundamental challenges in aligning LLMs with authentic human ideological positions through straightforward prompting techniques. This research contributes valuable insights for computational social scientists and AI ethicists employing LLMs as human simulacra in social science research, offering critical guidance on the limitations and potential methodological improvements for using language models in ideological representation tasks.

## Questionnaires
The current experimental iteration contains two questionnaires about moral dimensions:

### Moral Foundation Questionnaire (MFQ)
The Moral Foundation Questionnaire (MFQ) is a 32-item assessment developed by Graham, Haidt, and Nosek [^mfq] to measure the degree to which individuals value five distinct moral foundations. These foundations include Care/Harm, which concerns the suffering of others and encompasses virtues of caring and kindness; Fairness/Cheating, which relates to proportional treatment, equality, rights, and justice; Loyalty/Betrayal, which addresses obligations of group membership, such as patriotism and self-sacrifice for the group; Authority/Subversion, which involves concerns related to social order and the obligations of hierarchical relationships; and Sanctity/Degradation, which focuses on concerns about physical and spiritual purity, including disgust reactions to violations. The MFQ is structured into two main sections. The first section, Moral Relevance, consists of 15 items that assess how relevant different considerations are when making moral judgments. The second section, Moral Judgments, also contains 15 items and assesses agreement with specific moral statements. Each item in the questionnaire is rated on a 6-point scale, and scores are calculated for each of the five moral foundations.

### Moral Foundation Questionnaire 2 (MFQ-2)
The Moral Foundation Questionnaire 2 (MFQ-2) [^mfq2] represents an updated and refined version of the original MFQ that expands the theoretical framework. In addition to including the original five foundations from the MFQ, it adds Liberty/Oppression as a sixth foundation, which addresses concerns about individual freedom and resistance to domination. The MFQ-2 also employs a more refined measurement approach with improved psychometric properties. The MFQ-2 consists of 36 items and features more precise item wording to reduce ambiguity. It offers enhanced reliability and validity compared to the original MFQ and has been designed for broader cross-cultural applicability, making it a more robust tool for assessing moral foundations across diverse populations.

### Humor Styles Questionnaire (HSQ)
The Humor Styles Questionnaire (HSQ) was developed by Martin et al. [^hsq] to assess individual differences in uses of humor and their relation to psychological well-being. The HSQ consists of 32 statements rated on a five-point scale where 1=Never or very rarely true, 2=Rarely true, 3=Sometimes true, 4=Often true, 5=Very often or always true. The questionnaire measures four distinct humor styles: **Affiliative Humor:** Tendency to use humor to facilitate relationships and reduce interpersonal tensions. **Self-enhancing Humor:** Using humor to cope with stress and maintain a positive outlook. **Aggressive Humor:** Using humor to criticize or manipulate others through teasing or ridicule. **Self-defeating Humor:** Tendency to use excessively self-disparaging humor to gain approval. The HSQ provides scores for each of these four dimensions, calculated as the mean of the relevant items (with some items reverse-scored).

### Big Five Personality Inventory (BIG5)
This personality inventory assesses the five major dimensions of personality commonly known as the "Big Five" [^big5]. The questionnaire consists of 50 items rated on a five-point scale where 1=Disagree, 3=Neutral, 5=Agree. The five personality dimensions measured are: **Extraversion (E):** Sociability, assertiveness, and positive emotionality, **Neuroticism (N):** Emotional instability and tendency to experience negative emotions, **Agreeableness (A):** Prosocial attitudes, altruism, and cooperativeness, **Conscientiousness (C):** Organization, self-discipline, and achievement-orientation, **Openness to Experience (O):** Intellectual curiosity, creativity, and preference for novelty. Each dimension is assessed through 10 items, with some items reverse-scored.

## Implementation

 - `data/`: Directory containing questionnaires and human baseline survey data
 - `experiments/`: Directory containing experiments with synthetic survey data and reporting
 - `src/llm_questionnaires`: Directory containing package source files
  - `test`: Directory containing module tests and examples code

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
// used: Moral Foundation Questionnaire (MFQ)
@article{munker2025political,
  title={Political Bias in LLMs: Unaligned Moral Values in Agent-centric Simulations},
  author={M{\"u}nker, Simon},
  journal={Journal for Language Technology and Computational Linguistics},
  volume={38},
  number={2},
  pages={125--138},
  year={2025}
}

// used: Moral Foundation Questionnaire 2 (MFQ-2)
@inproceedings{munker2025cultural,
  title={Cultural Bias in Large Language Models: Evaluating AI Agents through Moral Questionnaires},
  author={M{\"u}nker, Simon},
  booktitle={Proceedings of 0 th Moral and Legal AI Alignment Symposium},
  pages={61},
  year={2025}
}
```


[^mfq]: Graham, J., Haidt, J., & Nosek, B. A. (2009). Liberals and conservatives rely on different sets of moral foundations. Journal of personality and social psychology, 96(5), 1029.
[^mfq2]: Atari, M., Haidt, J., Graham, J., Koleva, S., Stevens, S. T., & Dehghani, M. (2023). Morality beyond the WEIRD: How the nomological network of morality varies across cultures. Journal of Personality and Social Psychology, 125(5), 1157.
[^hsq]: Martin, R. A., Puhlik-Doris, P., Larsen, G., Gray, J., & Weir, K. (2003). Individual differences in uses of humor and their relation to psychological well-being: Development of the Humor Styles Questionnaire. Journal of research in personality, 37(1), 48-75.
[^big5]: Goldberg, L. R., Johnson, J. A., Eber, H. W., Hogan, R., Ashton, M. C., Cloninger, C. R., & Gough, H. G. (2006). The international personality item pool and the future of public-domain personality measures. Journal of Research in personality, 40(1), 84-96.