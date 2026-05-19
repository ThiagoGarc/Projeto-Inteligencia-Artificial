#Autores:
#Mateo Zanette RA: 10417980
#Thiago Felipe Garcia RA: 10414699
#Mateus Mendes Cabral RA: 10417820

#Última atualização (19/05):
#-Unificação dos arquivos dos códigos contendo os algoritmos de mineração de regras
#-Adição do gráfico "D Chord" como experimentação para disposição visual das regras



import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
pd.set_option('display.max_columns', None)
import warnings


df = pd.read_csv(r"C:\Users\thiag\OneDrive\Área de Trabalho\Faculdade\Projeto\test.csv")
df = df.sample(n=1000, random_state=42)

#Parâmetros
suporte = 0.6
confianca = 0.7
lift = 1.1


#Idade(Age)
df["Age_14"] = df["Age"].apply(lambda x: 1 if x == 14 else 0)
df["Age_15"] = df["Age"].apply(lambda x: 1 if x == 15 else 0)
df["Age_16"] = df["Age"].apply(lambda x: 1 if x == 16 else 0)
df["Age_17"] = df["Age"].apply(lambda x: 1 if x == 17 else 0)
df["Age_18"] = df["Age"].apply(lambda x: 1 if x == 18 else 0)
df.drop(columns=["Age"], inplace=True)


#Ano(Grade)
df["Grade_9"]  = df["Grade"].apply(lambda x: 1 if x == 9 else 0)
df["Grade_10"] = df["Grade"].apply(lambda x: 1 if x == 10 else 0)
df["Grade_11"] = df["Grade"].apply(lambda x: 1 if x == 11 else 0)
df["Grade_12"] = df["Grade"].apply(lambda x: 1 if x == 12 else 0)
df.drop(columns=["Grade"], inplace=True)


#Gênero(Gender)
df["Female"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)
df["Male"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)
df.drop(columns=["Gender"], inplace=True)


#Raça(Race)
df["Race_Asian"] = df["Race"].apply(lambda x: 1 if x == "Asian" else 0)
df["Race_Black"] = df["Race"].apply(lambda x: 1 if x == "Black" else 0)
df["Race_White"] = df["Race"].apply(lambda x: 1 if x == "White" else 0)
df["Race_Hispanic"] = df["Race"].apply(lambda x: 1 if x == "Hispanic" else 0)
df["Race_Other"] = df["Race"].apply(lambda x: 1 if x == "Other" else 0)
df["Race_Two_or_more"] = df["Race"].apply(lambda x: 1 if x == "Two-or-more" else 0)
df.drop(columns=["Race"], inplace=True)


#NívelSocieconomico(SES_Quartile)
df["SES_1"] = df["SES_Quartile"].apply(lambda x: 1 if x == 1 else 0)
df["SES_2"] = df["SES_Quartile"].apply(lambda x: 1 if x == 2 else 0)
df["SES_3"] = df["SES_Quartile"].apply(lambda x: 1 if x == 3 else 0)
df["SES_4"] = df["SES_Quartile"].apply(lambda x: 1 if x == 4 else 0)
df.drop(columns=["SES_Quartile"], inplace=True)


#EducaçãoParental(ParentalEducation)
df["PE_HS"] = df["ParentalEducation"].apply(lambda x: 1 if x == "HS" else 0)
df["PE_less_HS"] = df["ParentalEducation"].apply(lambda x: 1 if x == "<HS" else 0)
df["PE_Bachelors_plus"] = df["ParentalEducation"].apply(lambda x: 1 if x == "Bachelors+" else 0)
df["PE_SomeCollege"] = df["ParentalEducation"].apply(lambda x: 1 if x == "SomeCollege" else 0)
df.drop(columns=["ParentalEducation"], inplace=True)


#TipoEscola(SchoolType)
df["School_Public"] = df["SchoolType"].apply(lambda x: 1 if x == "Public" else 0)
df["School_Private"] = df["SchoolType"].apply(lambda x: 1 if x == "Private" else 0)
df.drop(columns=["SchoolType"], inplace=True)


#Localização(Locale)
df["Locale_Rural"] = df["Locale"].apply(lambda x: 1 if x == "Rural" else 0)
df["Locale_Town"] = df["Locale"].apply(lambda x: 1 if x == "Town" else 0)
df["Locale_Suburban"] = df["Locale"].apply(lambda x: 1 if x == "Suburban" else 0)
df["Locale_City"] = df["Locale"].apply(lambda x: 1 if x == "City" else 0)
df.drop(columns=["Locale"], inplace=True)


#Notas
#Arredondamento
df["TestScore_Math"] = df["TestScore_Math"].round(1)
df["TestScore_Reading"] = df["TestScore_Reading"].round(1)
df["TestScore_Science"] = df["TestScore_Science"].round(1)
df["GPA"] = df["GPA"].round(1)
#Matemática
df["Math_Low"] = df["TestScore_Math"].apply(lambda x: 1 if x < 60 else 0)
df["Math_Medium"] = df["TestScore_Math"].apply(lambda x: 1 if 60 <= x < 80 else 0)
df["Math_High"] = df["TestScore_Math"].apply(lambda x: 1 if x >= 80 else 0)
#Leitura
df["Reading_Low"] = df["TestScore_Reading"].apply(lambda x: 1 if x < 60 else 0)
df["Reading_Medium"] = df["TestScore_Reading"].apply(lambda x: 1 if 60 <= x < 80 else 0)
df["Reading_High"] = df["TestScore_Reading"].apply(lambda x: 1 if x >= 80 else 0)
#Ciência
df["Science_Low"] = df["TestScore_Science"].apply(lambda x: 1 if x < 60 else 0)
df["Science_Medium"] = df["TestScore_Science"].apply(lambda x: 1 if 60 <= x < 80 else 0)
df["Science_High"] = df["TestScore_Science"].apply(lambda x: 1 if x >= 80 else 0)
#GPA
df["GPA_1"] = df["GPA"].apply(lambda x: 1 if 1<= x <2 else 0)
df["GPA_2"] = df["GPA"].apply(lambda x: 1 if 2<= x <3 else 0)
df["GPA_3"] = df["GPA"].apply(lambda x: 1 if 3<= x <4 else 0)
df["GPA_4"] = df["GPA"].apply(lambda x: 1 if 4<= x <5 else 0)
df.drop(columns=["TestScore_Math", "TestScore_Reading", "TestScore_Science", "GPA"], inplace=True)


#HorasEstudo(StudyHours)
df["StudyHours"] = df["StudyHours"].round(1)
df["StudyHours_0"] = df["StudyHours"].apply(lambda x: 1 if 0< x <1 else 0)
df["StudyHours_1"] = df["StudyHours"].apply(lambda x: 1 if 1<= x <2 else 0)
df["StudyHours_2"] = df["StudyHours"].apply(lambda x: 1 if x>=2 else 0)
df.drop(columns=["StudyHours"], inplace=True)


#Frequência(AttendanceRate)
df["AttendanceRate"] = (df["AttendanceRate"] * 100).round(1)
df["AttendanceRate_70"] = df["AttendanceRate"].apply(lambda x: 1 if 70<= x <80 else 0)
df["AttendanceRate_80"] = df["AttendanceRate"].apply(lambda x: 1 if 80<= x <90 else 0)
df["AttendanceRate_90"] = df["AttendanceRate"].apply(lambda x: 1 if 90<= x <100 else 0)
df["AttendanceRate_100"] = df["AttendanceRate"].apply(lambda x: 1 if x==100 else 0)
df.drop(columns=["AttendanceRate"], inplace=True)


#TempoLivre(FreeTime)
df["FreeTime_1"] = df["FreeTime"].apply(lambda x: 1 if x == 1 else 0)
df["FreeTime_2"] = df["FreeTime"].apply(lambda x: 1 if x == 2 else 0)
df["FreeTime_3"] = df["FreeTime"].apply(lambda x: 1 if x == 3 else 0)
df["FreeTime_4"] = df["FreeTime"].apply(lambda x: 1 if x == 4 else 0)
df["FreeTime_5"] = df["FreeTime"].apply(lambda x: 1 if x == 5 else 0)
df.drop(columns=["FreeTime"], inplace=True)


#Passeio(GoOut)
df["GoOut_1"] = df["GoOut"].apply(lambda x: 1 if x == 1 else 0)
df["GoOut_2"] = df["GoOut"].apply(lambda x: 1 if x == 2 else 0)
df["GoOut_3"] = df["GoOut"].apply(lambda x: 1 if x == 3 else 0)
df["GoOut_4"] = df["GoOut"].apply(lambda x: 1 if x == 4 else 0)
df["GoOut_5"] = df["GoOut"].apply(lambda x: 1 if x == 5 else 0)
df.drop(columns=["GoOut"], inplace=True)


def codigo_apriori(df, confianca, suporte, lift):   
    df = df.astype(bool)
    frequent_itemsets = apriori(
        df,
        min_support=suporte,
        use_colnames=True,
        max_len=2
    )
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=confianca
    )
    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]
    #Excel
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
    rules.to_excel(r"C:\Users\thiag\OneDrive\Área de Trabalho\Faculdade\Projeto\regrasAPRIORI.xlsx", index=False)
    print("Regras APRIORI salvas.")


def eclat(df, confianca, suporte, lift):
    df = df.astype(bool)
    min_support = suporte
    min_confidence = confianca
    n_transactions = len(df)
    tid_lists = {}
    for col in df.columns:
        tids = set(df.index[df[col]])
        tid_lists[col] = tids

    #Itemsets de tamanho igual a 1
    frequent_items = {}
    for item, tids in tid_lists.items():
        support = len(tids) / n_transactions
        if support >= min_support:
            frequent_items[frozenset([item])] = tids

    #Itemsets de tamanho igual a 2
    frequent_pairs = {}
    items = list(frequent_items.keys())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item1 = items[i]
            item2 = items[j]
            tids = frequent_items[item1].intersection(frequent_items[item2])
            support = len(tids) / n_transactions
            if support >= min_support:
                frequent_pairs[frozenset(item1.union(item2))] = tids

    #Regras
    rules_list = []
    for pair, tids in frequent_pairs.items():
        items = list(pair)
        A = frozenset([items[0]])
        B = frozenset([items[1]])
        support_AB = len(tids) / n_transactions
        support_A = len(frequent_items[A]) / n_transactions
        support_B = len(frequent_items[B]) / n_transactions
        confidence_A_B = support_AB / support_A
        confidence_B_A = support_AB / support_B
        lift_A_B = confidence_A_B / support_B
        lift_B_A = confidence_B_A / support_A

        if lift_A_B >= min_confidence:
            rules_list.append({
                "antecedents": list(A)[0],
                "consequents": list(B)[0],
                "support": support_AB,
                "confidence": confidence_A_B,
                "lift": lift_A_B
            })
        if lift_B_A >= min_confidence:
            rules_list.append({
                "antecedents": list(B)[0],
                "consequents": list(A)[0],
                "support": support_AB,
                "confidence": confidence_B_A,
                "lift": lift_B_A
            })

    rules_eclat = pd.DataFrame(
        rules_list,
        columns=["antecedents", "consequents", "support", "confidence", "lift"]
    )
    #Excel
    rules_eclat.to_excel(
        r"C:\Users\thiag\OneDrive\Área de Trabalho\Faculdade\Projeto\regrasECLAT.xlsx",
        index=False
    )
    print("Regras ECLAT salvas.")


def fp_growth(df, confianca, suporte, lift):
    gpa_columns = [
        "GPA_1", "GPA_2", "GPA_3", "GPA_4",
        "Locale_Rural", "Locale_Town", "Locale_Suburban", "Locale_City",
        "FreeTime_1_2", "FreeTime_3_4", "FreeTime_5",
        "SchoolType_Public", "SchoolType_Private",
        "AttendanceRate_70", "AttendanceRate_80", "AttendanceRate_90", "AttendanceRate_100",
        "GoOut_1_2", "GoOut_3_4", "GoOut_5",
    ]

    df_gpa = df[gpa_columns].astype(bool)
    frequent_itemsets = fpgrowth(
        df_gpa,
        min_support=suporte,
        use_colnames=True
    )
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=confianca
    )

    rules = rules[rules["lift"] >= lift]
    gpa_antecedents = ['GPA_1', 'GPA_2', 'GPA_3', 'GPA_4']
    rules_gpa_as_antecedents = rules[
        rules['antecedents'].apply(lambda x: any(item in x for item in gpa_antecedents))
    ]

    rules_gpa_as_consequents = rules[
        rules['consequents'].apply(lambda x: any(item in x for item in gpa_antecedents))
    ]

    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

    #Excel
    rules.to_excel(r"C:\Users\thiag\OneDrive\Área de Trabalho\Faculdade\TCC\regrasFPGROWTH.xlsx",index=False)
    print("Regras FP-GROWTH salvas.")


print("(0) - APRIORI")
print("(1) - ECLAT")
print("(2) - FP-GROWTH")
print("(3) - TODOS")

escolha = int(input(">>> "))
print("")

if (escolha == 0):
    codigo_apriori(df, confianca, suporte, lift)
elif (escolha == 1):
    eclat(df, confianca, suporte, lift)
elif (escolha == 2):
    fp_growth(df, confianca, suporte, lift)
elif (escolha == 3):
    codigo_apriori(df, confianca, suporte, lift)
    eclat(df, confianca, suporte, lift)
    fp_growth(df, confianca, suporte, lift)
print("")