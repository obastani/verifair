#include <iostream>
#include <math.h>
#include <random>
#include <list>
#include <ctime>

class Sampler {
public:
  virtual int sample() = 0;
};

class DT14indSampler : public Sampler {
public:
  DT14indSampler(int flag) {
    // Sex
    double sex_params[] = {0.3307, 0.6693};
    sex_distribution = std::discrete_distribution<int>(std::begin(sex_params), std::end(sex_params));

    // Relationship
    double relationship_params[] = {0.0481, 0.1557, 0.4051, 0.2551, 0.0301, 0.1059};
    relationship_distribution = std::discrete_distribution<int>(std::begin(relationship_params), std::end(relationship_params));

    // Flag
    this->flag = flag;
  }
  
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample() {
    while (true) {
      int val = sample_with_error();
      if (val != -1) {
	return val;
      }
    }
  }

private:
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample_with_error() {
    // features
    int sex, relationship;
    double age;

    // sample feature
    age = gaussian(38.5816, 13.6404325445);
    sex = sex_distribution(generator);
    relationship = relationship_distribution(generator);

    // rejection sampling
    if (sex != flag) {
      return -1;
    }
    if (age <= 18) {
      return -1;
    }

    // classifier
    int t;
    if (relationship == 0) {
      t = 1;
    } else if (relationship == 1) {
      if (age < 21.5) {
	t = 1;
      } else {
	if (age < 47.5) {
	  t = 1;
	} else {
	  t = 0;
	}
      }
    } else if (relationship == 2) {
      t = 1;
    } else if (relationship == 3) {
      if (age < 50.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else if (relationship == 4) {
      if (age < 49.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else {
      t = 1;
    }
    return 1 - t;
  }
  
  double gaussian(double mean, double std) {
    return mean + std * normal_distribution(generator);
  }
  
  std::default_random_engine generator;
  std::discrete_distribution<int> sex_distribution;
  std::discrete_distribution<int> relationship_distribution;
  std::normal_distribution<double> normal_distribution;
  int flag;
};
  
class DT14BNSampler : public Sampler {
public:
  DT14BNSampler(int flag) {
    // Sex
    double sex_params[] = {0.3307, 0.6693};
    sex_distribution = std::discrete_distribution<int>(std::begin(sex_params), std::end(sex_params));

    // Relationship 0
    double relationship_0_params[] = {0.0491, 0.1556, 0.4012, 0.2589, 0.0294, 0.1058};
    relationship_0_distribution = std::discrete_distribution<int>(std::begin(relationship_0_params), std::end(relationship_0_params));

    // Relationship 1
    double relationship_1_params[] = {0.0416, 0.1667, 0.4583, 0.2292, 0.0166, 0.0876};
    relationship_1_distribution = std::discrete_distribution<int>(std::begin(relationship_1_params), std::end(relationship_1_params));

    // Relationship 2
    double relationship_2_params[] = {0.0497, 0.1545, 0.4021, 0.2590, 0.0294, 0.1053};
    relationship_2_distribution = std::discrete_distribution<int>(std::begin(relationship_2_params), std::end(relationship_2_params));

    // Relationship 3
    double relationship_3_params[] = {0.0417, 0.1624, 0.3976, 0.2606, 0.0356, 0.1021};
    relationship_3_distribution = std::discrete_distribution<int>(std::begin(relationship_3_params), std::end(relationship_3_params));

    // Flag
    this->flag = flag;
  }
  
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample() {
    while (true) {
      int val = sample_with_error();
      if (val != -1) {
	return val;
      }
    }
  }

private:
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample_with_error() {
    // features
    int sex, relationship;
    double capital_gain, age;

    // sample feature
    sex = sex_distribution(generator);
    if (sex == 0) {
      capital_gain = gaussian(568.4105, 4924.262944116612);
      if (capital_gain < 7298.0) {
	age = gaussian(38.4208, 13.59834916451258);
	relationship = relationship_0_distribution(generator);
      } else {
	age = gaussian(38.8125, 13.910132997207468);
	relationship = relationship_1_distribution(generator);
      }
    } else {
      capital_gain = gaussian(1329.3700, 8326.312094835264);
      if (capital_gain < 5178.0) {
	age = gaussian(38.6361, 13.683694676511896);
	relationship = relationship_2_distribution(generator);
      } else {
	age = gaussian(38.2668, 13.684834672000974);
	relationship = relationship_3_distribution(generator);
      }
    }

    // rejection sampling
    if (sex != flag) {
      return -1;
    }
    if (age <= 18) {
      return -1;
    }
    
    // classifier
    int t;
    if (relationship == 0) {
      t = 1;
    } else if (relationship == 1) {
      if (age < 21.5) {
	t = 1;
      } else {
	if (age < 47.5) {
	  t = 1;
	} else {
	  t = 0;
	}
      }
    } else if (relationship == 2) {
      t = 1;
    } else if (relationship == 3) {
      if (age < 50.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else if (relationship == 4) {
      if (age < 49.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else {
      t = 1;
    }
    return 1 - t;
  }
  
  double gaussian(double mean, double std) {
    return mean + std * normal_distribution(generator);
  }
  
  std::default_random_engine generator;
  std::discrete_distribution<int> sex_distribution;
  std::discrete_distribution<int> relationship_0_distribution;
  std::discrete_distribution<int> relationship_1_distribution;
  std::discrete_distribution<int> relationship_2_distribution;
  std::discrete_distribution<int> relationship_3_distribution;
  std::normal_distribution<double> normal_distribution;
  int flag;
};
  
class DT14BNcSampler : public Sampler {
public:
  DT14BNcSampler(int flag) {
    // Sex
    double sex_params[] = {0.3307, 0.6693};
    sex_distribution = std::discrete_distribution<int>(std::begin(sex_params), std::end(sex_params));

    // Education 0
    double education_0_params[] = {0.1638, 0.2308, 0.0354, 0.3230, 0.0173, 0.0321, 0.0412, 0.0156, 0.0200, 0.0112, 0.0528, 0.0050, 0.0290, 0.0119, 0.0092, 0.0017};
    education_0_distribution = std::discrete_distribution<int>(std::begin(education_0_params), std::end(education_0_params));

    // Education 1
    double education_1_params[] = {0.1916, 0.2000, 0.0500, 0.3542, 0.0208, 0.0125, 0.0375, 0.0125, 0.0292, 0.0042, 0.0541, 0.0000, 0.0250, 0.0042, 0.0042, 0.0000};
    education_1_distribution = std::discrete_distribution<int>(std::begin(education_1_params), std::end(education_1_params));

    // Education 2
    double education_2_params[] = {0.1670, 0.2239, 0.0358, 0.3267, 0.0159, 0.0320, 0.0426, 0.0155, 0.0198, 0.0121, 0.0518, 0.0047, 0.0287, 0.0125, 0.0096, 0.0014};
    education_2_distribution = std::discrete_distribution<int>(std::begin(education_2_params), std::end(education_2_params));

    // Education 3
    double education_3_params[] = {0.1569, 0.2205, 0.0417, 0.3071, 0.0255, 0.0302, 0.0409, 0.0155, 0.0178, 0.0147, 0.0619, 0.0062, 0.0317, 0.0139, 0.0139, 0.0016};
    education_3_distribution = std::discrete_distribution<int>(std::begin(education_3_params), std::end(education_3_params));

    // Relationship 0
    double relationship_0_params[] = {0.0491, 0.1556, 0.4012, 0.2589, 0.0294, 0.1058};
    relationship_0_distribution = std::discrete_distribution<int>(std::begin(relationship_0_params), std::end(relationship_0_params));

    // Relationship 1
    double relationship_1_params[] = {0.0416, 0.1667, 0.4583, 0.2292, 0.0166, 0.0876};
    relationship_1_distribution = std::discrete_distribution<int>(std::begin(relationship_1_params), std::end(relationship_1_params));

    // Relationship 2
    double relationship_2_params[] = {0.0497, 0.1545, 0.4021, 0.2590, 0.0294, 0.1053};
    relationship_2_distribution = std::discrete_distribution<int>(std::begin(relationship_2_params), std::end(relationship_2_params));

    // Relationship 3
    double relationship_3_params[] = {0.0417, 0.1624, 0.3976, 0.2606, 0.0356, 0.1021};
    relationship_3_distribution = std::discrete_distribution<int>(std::begin(relationship_3_params), std::end(relationship_3_params));

    // Flag
    this->flag = flag;
  }

  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample() {
    while (true) {
      int val = sample_with_error();
      if (val != -1) {
	return val;
      }
    }
  }

private:
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample_with_error() {
    // features
    int sex, education, relationship;
    double capital_gain, age, education_num;

    // sample feature
    sex = sex_distribution(generator);
    if (sex == 0) {
      capital_gain = gaussian(568.4105, 4924.26294412);
      if (capital_gain < 7298.0) {
	age = gaussian(38.4208, 13.5983491645);
	education = education_0_distribution(generator);
	education_num = gaussian(10.0827, 2.55139177705);
	relationship = relationship_0_distribution(generator);
      } else {
	age = gaussian(38.8125, 13.9101329972);
	education = education_1_distribution(generator);
	education_num = gaussian(10.1041, 2.48036287668);
	relationship = relationship_1_distribution(generator);
      }      
    } else {
      capital_gain = gaussian(1329.3700, 8326.31209484);
      if (capital_gain < 5178.0) {
	age = gaussian(38.6361, 13.6836946765);
	education = education_2_distribution(generator);
	education_num = gaussian(10.0817, 2.54638960098);
	relationship = relationship_2_distribution(generator);
      } else {
	age = gaussian(38.2668, 13.684834672);
	education = education_3_distribution(generator);
	education_num = gaussian(10.0974, 2.67942157937);
	relationship = relationship_3_distribution(generator);
      }
    }
    if (education_num > age) {
      age = education_num;
    }

    // rejection sampling
    if (sex != flag) {
      return -1;
    }
    if (age <= 18) {
      return -1;
    }

    // classifier
    int t;
    if (relationship == 0) {
      t = 1;
    } else if (relationship == 1) {
      if (age < 21.5) {
	t = 1;
      } else {
	if (age < 47.5) {
	  t = 1;
	} else {
	  t = 0;
	}
      }
    } else if (relationship == 2) {
      t = 1;
    } else if (relationship == 3) {
      if (age < 50.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else if (relationship == 4) {
      if (age < 49.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else {
      t = 1;
    }
    return 1 - t;
  }

  double gaussian(double mean, double std) {
    return mean + std * normal_distribution(generator);
  }
  
  std::default_random_engine generator;
  std::discrete_distribution<int> sex_distribution;
  std::discrete_distribution<int> education_0_distribution;
  std::discrete_distribution<int> education_1_distribution;
  std::discrete_distribution<int> education_2_distribution;
  std::discrete_distribution<int> education_3_distribution;
  std::discrete_distribution<int> relationship_0_distribution;
  std::discrete_distribution<int> relationship_1_distribution;
  std::discrete_distribution<int> relationship_2_distribution;
  std::discrete_distribution<int> relationship_3_distribution;
  std::normal_distribution<double> normal_distribution;
  int flag;
};

class DT44BNcSampler : public Sampler {
public:
  DT44BNcSampler(int flag) {
    // Sex
    double sex_params[] = {0.3307, 0.6693};
    sex_distribution = std::discrete_distribution<int>(std::begin(sex_params), std::end(sex_params));

    // Education 0
    double education_0_params[] = {0.1638, 0.2308, 0.0354, 0.3230, 0.0173, 0.0321, 0.0412, 0.0156, 0.0200, 0.0112, 0.0528, 0.0050, 0.0290, 0.0119, 0.0092, 0.0017};
    education_0_distribution = std::discrete_distribution<int>(std::begin(education_0_params), std::end(education_0_params));

    // Education 1
    double education_1_params[] = {0.1916, 0.2000, 0.0500, 0.3542, 0.0208, 0.0125, 0.0375, 0.0125, 0.0292, 0.0042, 0.0541, 0.0000, 0.0250, 0.0042, 0.0042, 0.0000};
    education_1_distribution = std::discrete_distribution<int>(std::begin(education_1_params), std::end(education_1_params));

    // Education 2
    double education_2_params[] = {0.1670, 0.2239, 0.0358, 0.3267, 0.0159, 0.0320, 0.0426, 0.0155, 0.0198, 0.0121, 0.0518, 0.0047, 0.0287, 0.0125, 0.0096, 0.0014};
    education_2_distribution = std::discrete_distribution<int>(std::begin(education_2_params), std::end(education_2_params));

    // Education 3
    double education_3_params[] = {0.1569, 0.2205, 0.0417, 0.3071, 0.0255, 0.0302, 0.0409, 0.0155, 0.0178, 0.0147, 0.0619, 0.0062, 0.0317, 0.0139, 0.0139, 0.0016};
    education_3_distribution = std::discrete_distribution<int>(std::begin(education_3_params), std::end(education_3_params));

    // Relationship 0
    double relationship_0_params[] = {0.0491, 0.1556, 0.4012, 0.2589, 0.0294, 0.1058};
    relationship_0_distribution = std::discrete_distribution<int>(std::begin(relationship_0_params), std::end(relationship_0_params));

    // Relationship 1
    double relationship_1_params[] = {0.0416, 0.1667, 0.4583, 0.2292, 0.0166, 0.0876};
    relationship_1_distribution = std::discrete_distribution<int>(std::begin(relationship_1_params), std::end(relationship_1_params));

    // Relationship 2
    double relationship_2_params[] = {0.0497, 0.1545, 0.4021, 0.2590, 0.0294, 0.1053};
    relationship_2_distribution = std::discrete_distribution<int>(std::begin(relationship_2_params), std::end(relationship_2_params));

    // Relationship 3
    double relationship_3_params[] = {0.0417, 0.1624, 0.3976, 0.2606, 0.0356, 0.1021};
    relationship_3_distribution = std::discrete_distribution<int>(std::begin(relationship_3_params), std::end(relationship_3_params));

    // Flag
    this->flag = flag;
  }

  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample() {
    while (true) {
      int val = sample_with_error();
      if (val != -1) {
	return val;
      }
    }
  }

private:
  // -1: error
  //  0: classified as no
  //  1: classified as yes
  int sample_with_error() {
    // features
    int sex, education, relationship;
    double capital_gain, age, education_num;

    // sample feature
    sex = sex_distribution(generator);
    if (sex == 0) {
      capital_gain = gaussian(568.4105, 4924.26294412);
      if (capital_gain < 7298.0) {
	age = gaussian(38.4208, 13.5983491645);
	education = education_0_distribution(generator);
	education_num = gaussian(10.0827, 2.55139177705);
	relationship = relationship_0_distribution(generator);
      } else {
	age = gaussian(38.8125, 13.9101329972);
	education = education_1_distribution(generator);
	education_num = gaussian(10.1041, 2.48036287668);
	relationship = relationship_1_distribution(generator);
      }      
    } else {
      capital_gain = gaussian(1329.3700, 8326.31209484);
      if (capital_gain < 5178.0) {
	age = gaussian(38.6361, 13.6836946765);
	education = education_2_distribution(generator);
	education_num = gaussian(10.0817, 2.54638960098);
	relationship = relationship_2_distribution(generator);
      } else {
	age = gaussian(38.2668, 13.684834672);
	education = education_3_distribution(generator);
	education_num = gaussian(10.0974, 2.67942157937);
	relationship = relationship_3_distribution(generator);
      }
    }
    if (education_num > age) {
      age = education_num;
    }

    // rejection sampling
    if (sex != flag) {
      return -1;
    }
    if (age <= 18) {
      return -1;
    }

    // classifier
    int t;
    if (relationship == 0) {
      if (education == 0
	  || (education >= 5 && education <= 7)
	  || education == 11
	  || education == 14) {
	t = 0;
      } else {
	t = 1;
      }
    } else if (relationship == 1) {
      if (capital_gain < 4718.5) {
	t = 1;
      } else {
	t = 0;
      }
    } else if (relationship == 2) {
      if (education == 0
	  || education == 5
	  || education == 11
	  || education == 14) {
	t = 0;
      } else {
	t = 1;
      }
    } else if (relationship == 3) {
      if (capital_gain < 8296.0) {
	t = 1;
      } else {
	t = 0;
      }
    } else if (relationship == 4) {
      t = 1;
    } else {
      if (capital_gain < 4668.5) {
	t = 1;
      } else {
	t = 0;
      }
    }
    return 1 - t;
  }

  double gaussian(double mean, double std) {
    return mean + std * normal_distribution(generator);
  }
  
  std::default_random_engine generator;
  std::discrete_distribution<int> sex_distribution;
  std::discrete_distribution<int> education_0_distribution;
  std::discrete_distribution<int> education_1_distribution;
  std::discrete_distribution<int> education_2_distribution;
  std::discrete_distribution<int> education_3_distribution;
  std::discrete_distribution<int> relationship_0_distribution;
  std::discrete_distribution<int> relationship_1_distribution;
  std::discrete_distribution<int> relationship_2_distribution;
  std::discrete_distribution<int> relationship_3_distribution;
  std::normal_distribution<double> normal_distribution;
  int flag;
};

double get_type(int n, double delta) {
  double nf = (double)n;
  double b = -log(delta / 24.0) / 1.8;
  double epsilon = sqrt((0.6 * log(log(nf)/log(1.1) + 1) + b) / nf);
  return epsilon;
}

// -1: no result
//  0: not fair
//  1: fair
int get_fairness_type(double c, double Delta, int n, double delta, double E_A, double E_B) {
  // Step 1: Get (epsilon, delta) values from the adaptive concentration inequality for the current number of samples
  double epsilon = get_type(n, delta);
  
  // Step 2: Check if |E_B| > epsilon_B
  if (abs(E_B) <= epsilon) {
    return -1;
  }

  // Step 3: Compute the type judgement for the fairness property (before the inequality)
  double E_fair = E_A / E_B - (1 - c);
  double epsilon_fair = epsilon / abs(E_B) + epsilon * (epsilon + abs(E_A)) / (abs(E_B) * (abs(E_B) - epsilon));
  double delta_fair = 2.0 * delta;

  // Temporary: Logging
  if (n%1000 == 0) {
    std::cout << n << ": " << E_A << ", " << E_B << ", " << E_fair << ", " << epsilon_fair << ", " << delta_fair << ", " << std::endl;
  }
  

  // Step 4: Check if fairness holds (with the inequality)
  if (E_fair - epsilon_fair >= -Delta) {
    return 1;
  }

  // Step 5: Check if fairness does not hold (with the inequality)
  if (E_fair + epsilon_fair <= Delta) {
    return 0;
  }

  // Step 6: Continue sampling
  return -1;
}

int verify(Sampler& model_A, Sampler& model_B, double c, double Delta, double delta, int n_max) {
  // Step 1: Initialization
  double nE_A = 0.0;
  double nE_B = 0.0;

  // Step 2: Iteratively sample and check whether fairness holds
  for (int i=0; i<n_max; ++i) {
    // Step 2a: Sample points
    int x = model_A.sample();
    int y = model_B.sample();

    // Step 2b: Update statistics
    nE_A += x;
    nE_B += y;
      
    // Step 2c: Normalized statistics
    int n = i + 1;
    double E_A = nE_A / n;
    double E_B = nE_B / n;
        
    // Step 2d: Get type judgement
    int t = get_fairness_type(c, Delta, n, delta, E_A, E_B);

    // Step 2e: Return if converged
    if (t != -1) {
      return t;
    }
  }

  return -1;
}
  
int main() {
  // Parameters
  DT44BNcSampler model0 = DT44BNcSampler(0);
  DT44BNcSampler model1 = DT44BNcSampler(1);
  double c = 0.15;
  double Delta = 0.0;
  double delta = 0.5 * 1.0e-10;
  int n_max = 10000000;

  // Verification
  clock_t start = clock();
  int result = verify(model1, model0, c, Delta, delta, n_max);
  clock_t end = clock();
  double time = (double) (end - start) / CLOCKS_PER_SEC;

  // Logging
  std::cout << "Result: " << result << std::endl;
  std::cout << "Time: " << time << std::endl;
  
  return 0;
}
