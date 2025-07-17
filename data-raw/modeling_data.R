## code to prepare `modeling_data` dataset goes here
library(cfbfastR)

cfbd_stats_season_advanced(year = 2024)

year <- 2024
team = 'Oregon'

get_ranked_wins_all_teams <- function(year) {
  # Preload all game stats and rankings for the season
  all_weekly_games <- purrr::map_dfr(1:13, function(week_num) {
    week_data <- cfbd_game_team_stats(year, week = week_num, division = "fbs")  |>
      dplyr::filter(conference %in% c("American Athletic",
                                      "FBS Indepenents", "Big 12",  "SEC", "ACC",
                                      "Conference USA", "Pac-12", "Southern",
                                      "Sun Belt", "Big Ten", "Mountain West",
                                      "Mid-American"))
    week_data$week <- week_num  # Add week for later filtering
    week_data
  })

  all_rankings <- purrr::map_dfr(1:13, function(week_num) {
    cfbd_rankings(year, week = week_num) |>
      dplyr::filter(poll == "AP Top 25") |>
      dplyr::mutate(week = week_num)
  })

  # Now compute ranked wins per team
  all_weekly_games |>
    dplyr::mutate(
      result = points > points_allowed,
      ranked_win = as.integer(result & opponent %in% all_rankings$school)
    ) |>
    dplyr::group_by(school)|>
    dplyr::summarise(ranked_wins = sum(ranked_win, na.rm = TRUE), .groups = "drop")
}


library(doParallel)

# Detect number of cores and create cluster
cl <- makeCluster(parallel::detectCores() - 1) # Leave 1 core free
registerDoParallel(cl)

full_data <- purrr::map_dfr(c(2014:2019, 2021:2024), function(year) {
  message("Processing year: ", year)
  Sys.sleep(1.5)

  team_stats <- cfbd_stats_season_team(year)
  adv_stats <- cfbd_stats_season_advanced(year)
  ranked_wins_df <- get_ranked_wins_all_teams(year)

  dplyr::inner_join(adv_stats, team_stats, by = "team") |>
    dplyr::left_join(ranked_wins_df, by = c("team" ="school")) |>
    dplyr::mutate(ranked_wins = tidyr::replace_na(ranked_wins, 0))
})
stopCluster(cl)
registerDoSEQ()




playoffs_2014 <- c("Alabama", "Oregon", "Florida State", "Ohio State")
playoffs_2015 <- c("Clemson", "Alabama", "Michigan State", "Oklahoma")
playoffs_2016 <- c("Alabama", "Clemson", "Ohio State", "Washington")
playoffs_2017 <- c("Clemson", "Georgia", "Oklahoma", "Alabama")
playoffs_2018 <- c("Alabama", "Clemson", "Notre Dame", "Oklahoma")
playoffs_2019 <- c("LSU", "Ohio State",  "Clemson", "Oklahoma")
playoffs_2021 <- c("Alabama", "Michigan", "Georgia", "Cincinnati")
playoffs_2022 <- c("Georgia", "Michigan", "TCU", "Ohio State")
playoffs_2023 <- c("Michigan", "Washington", "Texas", "Alabama")
playoffs_2024 <- c("Penn State", "Texas", "Notre Dame", "Ohio State")


full_data |>
  dplyr::mutate(
    playoff = dplyr::case_when(
      (year == 2014 & team %in% playoffs_2014) ~ 1,
      (year == 2015 & team %in% playoffs_2015) ~ 1,
      (year == 2016 & team %in% playoffs_2016) ~ 1,
      (year == 2017 & team %in% playoffs_2017) ~ 1,
      (year == 2018 & team %in% playoffs_2018) ~ 1,
      (year == 2019 & team %in% playoffs_2019) ~ 1,
      (year == 2021 & team %in% playoffs_2021) ~ 1,
      (year == 2024 & team %in% playoffs_2022) ~ 1,
      (year == 2023 & team %in% playoffs_2023) ~ 1,
      (year == 2024 & team %in% playoffs_2024) ~ 1,
      .default = 0
    )
  ) -> full_data

modeling_data <- full_data  %>%
  tidyr::replace_na(list(passes_intercepted = c(0),
                         passes_intercepted_yds = c(0),
                         passes_intercepted_TDs = 0)) |>
  mutate(playoff = factor(ifelse(playoff == 1, "True", "False"), levels = c("False","True")))

usethis::use_data(modeling_data, overwrite = TRUE)
