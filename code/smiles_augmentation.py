"""
SMILES数据增强模块

通过生成SMILES的随机等价表示来增加训练数据多样性。
同一分子可以有多种SMILES表示，例如苯环可以表示为：
- c1ccccc1
- c1cc(ccc1)
- c1ccc(cc1)
等等
"""

from rdkit import Chem, RDLogger


RDLogger.DisableLog('rdApp.warning')


def randomize_smiles(smiles, n_variants=1, rng=None, canonical_fallback=True):
    """生成SMILES的随机等价表示

    Args:
        smiles: 原始SMILES字符串
        n_variants: 生成的变体数量
        rng: 可选的random.Random实例，用于可复现实验
        canonical_fallback: 如果随机化未产生足够唯一变体，是否补充canonical SMILES

    Returns:
        list of randomized SMILES strings. 如果SMILES无效，返回原始SMILES的重复列表。
    """
    if n_variants <= 0:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * n_variants

    # RDKit的doRandom依赖内部随机状态。为保证同一个实验seed可复现，
    # 这里用rootedAtAtom随机选择起点，同时保留doRandom的遍历随机性。
    atom_count = mol.GetNumAtoms()
    variants = []
    seen = set()
    max_attempts = max(10, n_variants * 10)

    for _ in range(max_attempts):
        kwargs = {'doRandom': True}
        if rng is not None and atom_count > 0:
            kwargs['rootedAtAtom'] = rng.randrange(atom_count)

        randomized = Chem.MolToSmiles(mol, **kwargs)
        if randomized not in seen:
            variants.append(randomized)
            seen.add(randomized)
        if len(variants) >= n_variants:
            break

    if canonical_fallback and len(variants) < n_variants:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        while len(variants) < n_variants:
            variants.append(canonical)

    return variants


def augment_smiles(smiles, augmentation_prob=0.5, rng=None):
    """以一定概率对SMILES进行增强

    Args:
        smiles: 原始SMILES字符串
        augmentation_prob: 增强概率（0.0-1.0）
                          训练时以此概率应用随机化
        rng: 可选的random.Random实例

    Returns:
        增强后的SMILES（或原始SMILES）
    """
    if not 0.0 <= augmentation_prob <= 1.0:
        raise ValueError(f'augmentation_prob must be in [0, 1], got {augmentation_prob}')

    random_value = rng.random() if rng is not None else __import__('random').random()
    if random_value < augmentation_prob:
        variants = randomize_smiles(smiles, n_variants=1, rng=rng)
        return variants[0]
    return smiles


if __name__ == '__main__':
    # 测试代码
    test_smiles = [
        'c1ccccc1',  # 苯环
        'CC(C)C',    # 异丁烷
        'CCO',       # 乙醇
    ]

    print('SMILES增强测试：\n')
    for smiles in test_smiles:
        print(f'原始SMILES: {smiles}')
        variants = randomize_smiles(smiles, n_variants=3)
        for i, variant in enumerate(variants, 1):
            print(f'  变体{i}: {variant}')
        print()
