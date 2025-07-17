library(factoextra)
library(dplyr)


####################### PCA ###################################################

pca_data <- modeling_data %>%
  select(-c("playoff","conference.x", "conference.y","team")) %>%
  tidyr::replace_na(list(passes_intercepted = c(0), passes_intercepted_yds = c(0),
                    passes_intercepted_TDs = 0))

pca <- princomp(scale(pca_data))


fviz_eig(pca, addlabels = TRUE)

fviz_cos2(pca, choice = "var")

##########################

# Example correlation matrix
cor_matrix <- cor(pca_data)

# Set a threshold
threshold <- 0.8

# Zero out self-correlations (optional)
diag(cor_matrix) <- NA

# Get the upper triangle only (to avoid duplicates)
upper_tri <- upper.tri(cor_matrix)

# Apply threshold filter
high_cor_indices <- which(abs(cor_matrix) > threshold & upper_tri, arr.ind = TRUE)

# Create a named vector
high_cor_values <- cor_matrix[high_cor_indices]
names(high_cor_values) <- apply(high_cor_indices, 1, function(idx) {
  paste(rownames(cor_matrix)[idx[1]], colnames(cor_matrix)[idx[2]], sep = " - ")
})

# View the named vector
print(high_cor_values)

#REMOVE THE HIGHLY CORRELATED VARIABLES

pca_data %>% select(-c(off_ppa, off_total_ppa,
off_field_pos_avg_predicted_points,
off_havoc_total,
off_success_rate,
off_standard_downs_rate,
off_rushing_plays_ppa,
off_rushing_plays_rate,
off_drives,
def_drives,
def_ppa,
def_field_pos_avg_predicted_points,
def_havoc_total,
def_success_rate,
def_standard_downs_rate,
def_rushing_plays_ppa,
def_rushing_plays_rate,
def_passing_downs_total_ppa,
def_passing_plays_ppa,
season.x ,
season.y,
pass_atts,
rush_yds,
first_downs,
penalty_yds,
kick_returns,
def_standard_downs_ppa,
def_open_field_yds,
off_line_yds_total,
net_pass_yds, off_open_field_yds,
off_passing_plays_ppa,
def_standard_downs_success_rate,
off_passing_downs_explosiveness,
fumbles_lost,
off_explosiveness, def_passing_downs_explosiveness,
def_explosiveness, def_passing_downs_explosiveness, fourth_downs,
def_havoc_front_seven, fourth_down_convs, off_passing_plays_explosiveness,
off_rushing_plays_explosiveness, off_standard_downs_explosiveness, def_rushing_plays_explosiveness,
kick_return_yds, def_standard_downs_explosiveness, pass_comps, third_downs, penalties, punt_return_TDs,
def_havoc_db, off_havoc_front_seven )   ) -> test2


# Example correlation matrix
cor_matrix <- cor(test2)

# Set a threshold
threshold <- 0.85

# Zero out self-correlations (optional)
diag(cor_matrix) <- NA

# Get the upper triangle only (to avoid duplicates)
upper_tri <- upper.tri(cor_matrix)

# Apply threshold filter
high_cor_indices <- which(abs(cor_matrix) > threshold & upper_tri, arr.ind = TRUE)

# Create a named vector
high_cor_values <- cor_matrix[high_cor_indices]
names(high_cor_values) <- apply(high_cor_indices, 1, function(idx) {
  paste(rownames(cor_matrix)[idx[1]], colnames(cor_matrix)[idx[2]], sep = " - ")
})

# View the named vector
print(high_cor_values)

# pca <- princomp(scale(test2))
#
#
# fviz_eig(pca, addlabels = TRUE)
#
# fviz_cos2(pca, choice = "var")
