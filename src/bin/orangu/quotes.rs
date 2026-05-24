// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

pub enum QuoteModule {
    None,
    StarTrek,
    StarWars,
    MarcoPierreWhite,
    GordonRamsay,
    CalvinAndHobbes,
    All,
}

impl QuoteModule {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "star_trek" => Self::StarTrek,
            "star_wars" => Self::StarWars,
            "marco_pierre_white" => Self::MarcoPierreWhite,
            "gordon_ramsay" => Self::GordonRamsay,
            "calvin_and_hobbes" => Self::CalvinAndHobbes,
            "all" => Self::All,
            _ => Self::None,
        }
    }

    pub fn pick(&self, seed: u64) -> Option<&'static str> {
        let pool: &[&[&str]] = match self {
            Self::None => return None,
            Self::StarTrek => &[STAR_TREK],
            Self::StarWars => &[STAR_WARS],
            Self::MarcoPierreWhite => &[MARCO_PIERRE_WHITE],
            Self::GordonRamsay => &[GORDON_RAMSAY],
            Self::CalvinAndHobbes => &[CALVIN_AND_HOBBES],
            Self::All => &[
                STAR_TREK,
                STAR_WARS,
                MARCO_PIERRE_WHITE,
                GORDON_RAMSAY,
                CALVIN_AND_HOBBES,
            ],
        };
        let total: usize = pool.iter().map(|s| s.len()).sum();
        if total == 0 {
            return None;
        }
        let mut idx = seed as usize % total;
        for slice in pool {
            if idx < slice.len() {
                return Some(slice[idx]);
            }
            idx -= slice.len();
        }
        None
    }
}

static STAR_TREK: &[&str] = &[
    "Space: the final frontier. These are the voyages of the starship Enterprise.",
    "Live long and prosper.",
    "Make it so.",
    "Resistance is futile.",
    "He's dead, Jim.",
    "I'm a doctor, not a bricklayer.",
    "To boldly go where no man has gone before.",
    "Fascinating.",
    "Engage.",
    "The needs of the many outweigh the needs of the few.",
];

static STAR_WARS: &[&str] = &[
    "May the Force be with you.",
    "Do. Or do not. There is no try.",
    "I am your father.",
    "I've got a bad feeling about this.",
    "In my experience, there is no such thing as luck.",
    "The Force will be with you, always.",
    "That's no moon.",
    "Help me, Obi-Wan Kenobi. You're my only hope.",
    "It's a trap!",
    "Never tell me the odds.",
];

static MARCO_PIERRE_WHITE: &[&str] = &[
    "Cooking is the easy part. Doing it every day is the hard part.",
    "I cook for people, not for critics.",
    "The more you learn, the more you realize how little you know.",
    "Perfection is a lot of little things done well.",
    "Nature is the true artist; the chef is merely the cook.",
    "When you understand an ingredient, you understand the dish.",
    "Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for life.",
    "A tree without roots is just a piece of wood.",
    "If you are not extreme, then people will take shortcuts because they don't fear you.",
    "Strategy will compensate the talent. The talent will never compensate the strategy.",
    "Mother Nature is the true artist and our job as cooks is to allow her to shine.",
    "Success is born out of arrogance, but greatness comes from humility.",
    "Once you accept you are being judged by people who have less knowledge than yourself, then what's it worth?",
    "People who can give themselves every day. They're the people that I admire, they're real people.",
];

static GORDON_RAMSAY: &[&str] = &[
    "This chicken is so raw it's still asking why it crossed the road.",
    "A recipe has no soul. You, as the cook, must bring soul to the recipe.",
    "Cooking today is a young man's game — I don't give a bollocks what anyone says.",
    "I don't like looking back. I'm always constantly looking forward.",
    "Kitchens are hard environments and they form incredibly strong characters.",
    "I taught myself to cook — that is what makes me a truly self-made chef.",
    "The secret of a good restaurant: find great ingredients and don't ruin them.",
];

static CALVIN_AND_HOBBES: &[&str] = &[
    "It's a magical world, Hobbes, ol' buddy... let's go exploring!",
    "Sometimes I think the surest sign that intelligent life exists elsewhere in the universe is that none of it has tried to contact us.",
    "Reality continues to ruin my life.",
    "I go to school, but I never learn what I want to know.",
    "Life is like topography, Hobbes. There are summits of happiness and success, flat stretches of boring routine, and valleys of frustration and failure.",
    "I'm not dumb, I just have a command of thoroughly useless information.",
    "You can't just turn on creativity like a faucet. You have to be in the right mood. What mood is that? Last-minute panic.",
    "Weekends don't count unless you spend them doing something completely pointless.",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_returns_no_quote() {
        assert!(QuoteModule::None.pick(0).is_none());
        assert!(QuoteModule::None.pick(42).is_none());
    }

    #[test]
    fn each_module_returns_quotes() {
        assert!(QuoteModule::StarTrek.pick(0).is_some());
        assert!(QuoteModule::StarWars.pick(1).is_some());
        assert!(QuoteModule::MarcoPierreWhite.pick(2).is_some());
        assert!(QuoteModule::GordonRamsay.pick(3).is_some());
        assert!(QuoteModule::CalvinAndHobbes.pick(4).is_some());
    }

    #[test]
    fn all_covers_every_slot() {
        let total: usize = [
            STAR_TREK,
            STAR_WARS,
            MARCO_PIERRE_WHITE,
            GORDON_RAMSAY,
            CALVIN_AND_HOBBES,
        ]
        .iter()
        .map(|s| s.len())
        .sum();
        for seed in 0..total as u64 {
            assert!(QuoteModule::All.pick(seed).is_some());
        }
    }

    #[test]
    fn from_str_parses_all_variants() {
        assert!(matches!(QuoteModule::from_str("none"), QuoteModule::None));
        assert!(matches!(
            QuoteModule::from_str("star_trek"),
            QuoteModule::StarTrek
        ));
        assert!(matches!(
            QuoteModule::from_str("star_wars"),
            QuoteModule::StarWars
        ));
        assert!(matches!(
            QuoteModule::from_str("marco_pierre_white"),
            QuoteModule::MarcoPierreWhite
        ));
        assert!(matches!(
            QuoteModule::from_str("gordon_ramsay"),
            QuoteModule::GordonRamsay
        ));
        assert!(matches!(
            QuoteModule::from_str("calvin_and_hobbes"),
            QuoteModule::CalvinAndHobbes
        ));
        assert!(matches!(QuoteModule::from_str("all"), QuoteModule::All));
        assert!(matches!(
            QuoteModule::from_str("unknown"),
            QuoteModule::None
        ));
    }
}
