from sym_modeling.domains.fem.imports import *
from sym_modeling.domains.fem.data import *
from sym_modeling.domains.fem.methods.euclid.feature_library import *
from sym_modeling.domains.fem.operators.kinematics import *


def loadFemData(path, AD=True, noiseLevel=0.0, noiseType="displacement", denoisedDisplacements=None):
    """
    Load finite element data and add noise (optional).
    Note that the loaded finite element data might already be perturbed by noise.
    In that case, adding additional noise is not necessary.
    """
    print("\n-----------------------------------------------------")
    print("Loading data: ", path)

    if path[-1] == "/":
        path = path[0:-1]

    numNodesPerElement = 3

    df = pd.read_csv(path + "/output_nodes.csv", dtype=np.float64)
    x_nodes = df[["x", "y"]].values
    u_nodes = df[["ux", "uy"]].values

    if denoisedDisplacements is not None:
        u_nodes = np.asarray(denoisedDisplacements, dtype=float)

    bcs_nodes = df[["bcx", "bcy"]].round().astype(int).values
    dirichlet_nodes = bcs_nodes != 0

    if noiseType not in ("displacement", "strain"):
        raise ValueError("Incorrect noiseType argument!")

    noise_nodes = noiseLevel * np.random.randn(*u_nodes.shape)
    noise_nodes[dirichlet_nodes] = 0.0
    if noiseType == "displacement":
        u_nodes = u_nodes + noise_nodes
        print("Applying noise to displacements:", noiseLevel)

    numReactions = int(np.max(bcs_nodes))
    df = pd.read_csv(path + "/output_reactions.csv", dtype=np.float64)
    reactions = []
    for i in range(numReactions):
        reactions.append(Reaction(bcs_nodes == (i + 1), df["forces"][i]))

    df = pd.read_csv(path + "/output_elements.csv", dtype=np.float64)
    connectivity = []
    for i in range(numNodesPerElement):
        connectivity.append(df["node" + str(i + 1)].round().astype(int).to_numpy())

    P = None
    P_columns = ["Pxx", "Pxy", "Pyx", "Pyy"]
    if all(column in df.columns for column in P_columns):
        P = df[P_columns].values

    df = pd.read_csv(path + "/output_integrator.csv", dtype=np.float64)
    gradNa = []
    for i in range(numNodesPerElement):
        gradNa.append(
            df[["gradNa_node" + str(i + 1) + "_x", "gradNa_node" + str(i + 1) + "_y"]].values
        )
    qpWeights = df["qpWeight"].values

    u = []
    for i in range(numNodesPerElement):
        u.append(u_nodes[connectivity[i], :])

    dim = 2
    voigtMap = [[0, 1], [2, 3]]
    numElements = qpWeights.shape[0]
    F = np.zeros((numElements, 4), dtype=float)
    for a in range(numNodesPerElement):
        for i in range(dim):
            for j in range(dim):
                F[:, voigtMap[i][j]] += u[a][:, i] * gradNa[a][:, j]
    F[:, 0] += 1.0
    F[:, 3] += 1.0

    if noiseType == "strain":
        F = F + noiseLevel * np.random.randn(*F.shape)
        print("Applying noise to strains:", noiseLevel)

    J = computeJacobian(F)
    C = computeCauchyGreenStrain(F)
    I1, I2, I3 = computeStrainInvariants(C)
    dI1dF = computeStrainInvariantDerivatives(F, 1)
    dI2dF = computeStrainInvariantDerivatives(F, 2)
    dI3dF = computeStrainInvariantDerivatives(F, 3)

    featureSet = FeatureSet()
    featureSet.features = computeFeatures(I1, I2, I3)
    featureSet.d_features_dI1, featureSet.d_features_dI2, featureSet.d_features_dI3 = (
        computeFeatureDerivatives(I1, I2, I3)
    )

    dataset = FemDataset(
        path,
        x_nodes,
        u_nodes,
        dirichlet_nodes,
        reactions,
        connectivity,
        gradNa,
        qpWeights,
        F,
        J,
        C,
        I1,
        I2,
        I3,
        dI1dF,
        dI2dF,
        dI3dF,
        featureSet,
        P=P,
    )

    print("-----------------------------------------------------\n")
    return dataset


def load_fem_dataset(*args, **kwargs):
    return loadFemData(*args, **kwargs)
